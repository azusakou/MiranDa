import torch as th
import torch.nn as nn
import geoopt as gt

PROJ_EPS = 1e-5
EPS = 1e-15


def project_hyp_vec(x):
    # To make sure hyperbolic embeddings are inside the unit ball.
    norm = th.sum(x ** 2, dim=-1, keepdim=True)

    return x * (1. - PROJ_EPS) / th.clamp(norm, 1. - PROJ_EPS)


def asinh(x):
    return th.log(x + (x ** 2 + 1) ** 0.5)


def acosh(x):
    return th.log(x + (x ** 2 - 1) ** 0.5)


def atanh(x):
    return 0.5 * th.log((1 + x) / (1 - x))


def poinc_dist(u, v):
    m = mob_add(-u, v) + EPS
    atanh_x = th.norm(m, dim=-1, keepdim=True)
    dist_poincare = 2.0 * atanh(atanh_x)
    return dist_poincare


def euclid_dist(u, v):
    return th.norm(u - v, dim=-1, keepdim=True)


def mob_add(u, v):
    v = v + EPS

    norm_uv = 2 * th.sum(u * v, dim=-1, keepdim=True)
    norm_u = th.sum(u ** 2, dim=-1, keepdim=True)
    norm_v = th.sum(v ** 2, dim=-1, keepdim=True)

    denominator = 1 + norm_uv + norm_v * norm_u
    result = (1 + norm_uv + norm_v) / denominator * u + (1 - norm_u) / denominator * v

    return project_hyp_vec(result)


def mob_scalar_mul(r, v):
    v = v + EPS
    norm_v = th.norm(v, dim=-1, keepdim=True)
    nomin = th.tanh(r * atanh(norm_v))
    result = nomin / norm_v * v

    return project_hyp_vec(result)


def mob_mat_mul(M, x):
    x = project_hyp_vec(x)
    Mx = x.matmul(M)
    Mx_norm = th.norm(Mx + EPS, dim=-1, keepdim=True)
    x_norm = th.norm(x + EPS, dim=-1, keepdim=True)

    return project_hyp_vec(th.tanh(Mx_norm / x_norm * atanh(x_norm)) / Mx_norm * Mx)


def mob_mat_mul_d(M, x, d_ball):
    x = project_hyp_vec(x)
    Mx = x.view(x.shape[0], -1).matmul(M.view(M.shape[0] * d_ball, M.shape[0] * d_ball)).view(x.shape)
    Mx_norm = th.norm(Mx + EPS, dim=-1, keepdim=True)
    x_norm = th.norm(x + EPS, dim=-1, keepdim=True)

    return project_hyp_vec(th.tanh(Mx_norm / x_norm * atanh(x_norm)) / Mx_norm * Mx)


def lambda_x(x):
    return 2. / (1 - th.sum(x ** 2, dim=-1, keepdim=True))


def exp_map_x(x, v):
    v = v + EPS
    second_term = th.tanh(lambda_x(x) * th.norm(v) / 2) / th.norm(v) * v
    return mob_add(x, second_term)


def log_map_x(x, y):
    diff = mob_add(-x, y) + EPS
    return 2. / lambda_x(x) * atanh(th.norm(diff, dim=-1, keepdim=True)) / \
           th.norm(diff, dim=-1, keepdim=True) * diff


def exp_map_zero(v):
    v = v + EPS
    norm_v = th.norm(v, dim=-1, keepdim=True)
    result = th.tanh(norm_v) / norm_v * v

    return project_hyp_vec(result)


def log_map_zero(y):
    diff = project_hyp_vec(y + EPS)
    norm_diff = th.norm(diff, dim=-1, keepdim=True)
    return atanh(norm_diff) / norm_diff * diff


def mob_pointwise_prod(x, u):
    # x is hyperbolic, u is Euclidean
    x = project_hyp_vec(x + EPS)
    Mx = x * u
    Mx_norm = th.norm(Mx + EPS, dim=-1, keepdim=True)
    x_norm = th.norm(x, dim=-1, keepdim=True)

    result = th.tanh(Mx_norm / x_norm * atanh(x_norm)) / Mx_norm * Mx
    return project_hyp_vec(result)


class hyperDense(nn.Module):

    def __init__(self, in_features, out_features):
        super(hyperDense, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        k = (1 / in_features) ** 0.5
        self.w = gt.ManifoldParameter(gt.ManifoldTensor(in_features, out_features).uniform_(-k, k))
        self.b = gt.ManifoldParameter(gt.ManifoldTensor(out_features).zero_())

    def forward(self, inputs):
        hyp_b = exp_map_zero(self.b)

        wx = mob_mat_mul(self.w, inputs)

        return mob_add(wx, hyp_b)


class hyperRNN(nn.Module):

    def __init__(self, input_size, hidden_size, default_dtype=th.float64):
        super(hyperRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.default_dtype = default_dtype

        k = (1 / hidden_size) ** 0.5
        self.w = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k, k))
        self.u = gt.ManifoldParameter(gt.ManifoldTensor(input_size, hidden_size).uniform_(-k, k))
        self.b = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, manifold=gt.PoincareBall()).zero_())

    def transition(self, x, h):
        W_otimes_h = mob_mat_mul(self.w, h)
        U_otimes_x = mob_mat_mul(self.u, x)
        Wh_plus_Ux = mob_add(W_otimes_h, U_otimes_x)

        return mob_add(Wh_plus_Ux, self.b)

    def init_rnn_state(self, batch_size, hidden_size, cuda_device):
        return th.zeros((batch_size, hidden_size), dtype=self.default_dtype, device=cuda_device)

    def forward(self, inputs):
        hidden = self.init_rnn_state(inputs.shape[0], self.hidden_size, inputs.device)
        outputs = []
        for x in inputs.transpose(0, 1):
            hidden = self.transition(x, hidden)
            outputs += [hidden]
        return th.stack(outputs).transpose(0, 1)


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        k = (1 / hidden_size) ** 0.5
        self.w_z = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k, k))
        self.w_r = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k, k))
        self.w_h = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k, k))
        self.u_z = gt.ManifoldParameter(gt.ManifoldTensor(input_size, hidden_size).uniform_(-k, k))
        self.u_r = gt.ManifoldParameter(gt.ManifoldTensor(input_size, hidden_size).uniform_(-k, k))
        self.u_h = gt.ManifoldParameter(gt.ManifoldTensor(input_size, hidden_size).uniform_(-k, k))
        self.b_z = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, manifold=gt.PoincareBall()).zero_())
        self.b_r = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, manifold=gt.PoincareBall()).zero_())
        ghball = gt.Scaled(gt.ProductManifold((gt.Euclidean(), 32), (gt.PoincareBall(), 96)))
        self.b_h = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, manifold=gt.PoincareBall()).zero_())

    def transition(self, W, h, U, x, hyp_b):
        W_otimes_h = mob_mat_mul(W, h)
        U_otimes_x = mob_mat_mul(U, x)
        Wh_plus_Ux = mob_add(W_otimes_h, U_otimes_x)

        return mob_add(Wh_plus_Ux, hyp_b)

    def forward(self, hyp_x, hidden):
        z = self.transition(self.w_z, hidden, self.u_z, hyp_x, self.b_z)
        z = th.sigmoid(log_map_zero(z))

        r = self.transition(self.w_r, hidden, self.u_r, hyp_x, self.b_r)
        r = th.sigmoid(log_map_zero(r))

        r_point_h = mob_pointwise_prod(hidden, r)
        h_tilde = self.transition(self.w_h, r_point_h, self.u_r, hyp_x, self.b_h)
        # h_tilde = th.tanh(log_map_zero(h_tilde)) # non-linearity

        minus_h_oplus_htilde = mob_add(-hidden, h_tilde)
        new_h = mob_add(hidden, mob_pointwise_prod(minus_h_oplus_htilde, z))

        return new_h


class hyperGRU(nn.Module):

    def __init__(self, input_size, hidden_size, default_dtype=th.float32):
        super(hyperGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.default_dtype = default_dtype

        self.gru_cell = GRUCell(input_size, hidden_size)

    def init_gru_state(self, batch_size, hidden_size, cuda_device):
        return th.zeros((batch_size, hidden_size), dtype=self.default_dtype, device=cuda_device)

    def forward(self, inputs):
        hidden = self.init_gru_state(inputs.shape[0], self.hidden_size, inputs.device)
        outputs = []
        for x in inputs.transpose(0, 1):
            hidden = self.gru_cell(x, hidden)
            outputs += [hidden]
        return th.stack(outputs).transpose(0, 1)