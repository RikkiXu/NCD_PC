import torch
import torch.nn.functional as F
import torch.nn as nn


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, logits, queue=None, use_gt=False):

        logits = logits.to(torch.float64)
        Q = torch.exp(logits / self.epsilon).t()

        B = Q.shape[1]
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            if use_gt:
                K2 = torch.tensor([[0.6000, 0.2000, 0.1500, 0.0500]]).cuda()
                Q = (Q.t() * K2).t()
            else:
                Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        to_ret = Q.t() if queue is None else Q.t()[:-queue.shape[0]]

        return to_ret


class SinkhornKnopp_im(torch.nn.Module):
    def __init__(self, split=0, num_iters=3, epsilon=0.05, num_unlabel=3):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.w = torch.nn.Parameter(torch.ones(1, num_unlabel) / num_unlabel)
        self.w0 = torch.ones(1, num_unlabel).cuda() / num_unlabel
        self.K2 = F.softmax(self.w, dim=1)
        self.K2 = self.K2.cuda()
        self.num_unlabel = num_unlabel

    def get_w(self):
        return torch.log_softmax(self.w, dim=1)

    def get_w0(self):
        return torch.softmax(self.w0, dim=1)

    def forward(self, logits, split=0, mode='default', queue=None, return_logit=False):
        logits = logits.to(torch.float64)
        Q = torch.exp(logits / self.epsilon).t()
        B = Q.shape[1]
        sum_Q = torch.sum(Q)
        Q /= sum_Q
        self.K2 = F.softmax(self.w, dim=1).cuda()
        for it in range(self.num_iters):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q = (Q.t() * self.K2).t()
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B
        to_ret = Q.t() if queue is None else Q.t()[:-queue.shape[0]]
        if return_logit:
            return to_ret, logits
        else:
            return to_ret


class Balanced_sinkhorn(torch.nn.Module):
    def __init__(self, split, num_iters=3, epsilon=0.05, lr_w=0.1, num_outer_iters=10, gamma=5, num_unlabel=3):
        super().__init__()
        self.sk = SinkhornKnopp_im(num_iters=num_iters, epsilon=epsilon, split=split, num_unlabel=num_unlabel).cuda()
        self.opt = torch.optim.SGD(params=self.sk.parameters(), lr=lr_w, momentum=0.99)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.num_outer_iters = num_outer_iters
        self.gamma = gamma

    def forward(self, features, head):
        features = features.detach()
        # self.sk.ini()

        for it in range(self.num_outer_iters):
            Q, preds = self.sk(features, head, return_logit=True)
            loss = - torch.mean(torch.sum(Q * preds, dim=1))
            input = self.sk.get_w().t().cuda()
            target = self.sk.get_w0().t().cuda()
            reg = self.kl_loss(input, target)
            total_loss = loss + self.gamma * reg
            self.opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sk.parameters(), 1)
            self.opt.step()
        return Q


class Balanced_sinkhorn_ce(torch.nn.Module):
    def __init__(self, split, num_iters=3, epsilon=0.05, lr_w=0.01, num_outer_iters=10, lam=5, gamma_bound=0.05,
                 num_unlabel=3, path=None, log=None, gamma=None):
        super().__init__()
        self.sk = SinkhornKnopp_im(num_iters=num_iters, epsilon=epsilon, split=split, num_unlabel=num_unlabel).cuda()
        self.opt = torch.optim.SGD(params=self.sk.parameters(), lr=lr_w, momentum=0.9)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.num_outer_iters = num_outer_iters
        self.path = path
        self.lr_w = lr_w
        self.split = split
        self.lam = lam
        self.gamma = gamma

    def forward(self, features, head, epoch):

        features = features.detach()
        features = features.detach()
        head = head.detach()
        features = torch.nn.functional.normalize(features, dim=1, p=2)
        head = torch.nn.functional.normalize(head, dim=0, p=2)
        logits = features @ head

        last_w = torch.zeros_like(self.sk.K2)
        for it in range(self.num_outer_iters):
            Q, preds = self.sk(logits, return_logit=True)
            prob = F.softmax(preds / 0.1, dim=1)
            loss = - torch.mean(torch.sum(Q * torch.log(prob), dim=1))
            input = self.sk.get_w().t().cuda()
            target = self.sk.get_w0().t().cuda()
            reg = self.kl_loss(input, target)

            # if self.detach == None:
            #    self.gamma = torch.exp(self.lam * loss.detach()) - 1
            # else:
            # self.gamma = (self.lam * loss.detach() + self.gamma_bound) * self.ds**epoch
            # self.gamma =(torch.exp(self.lam * loss.detach()) - 1) / (epoch+1)
            total_loss = loss + self.gamma * reg
            self.opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sk.parameters(), 1)
            self.opt.step()

            if torch.norm(self.sk.K2 - last_w) < 1e-4:
                break
            last_w = self.sk.K2
        return Q, loss.item(), reg.item()


class SemiSinkhornKnopp(torch.nn.Module):
    """
    naive SinkhornKnopp algorithm for semi-relaxed optimal transport, one side is equality constraint, the other side is KL divergence constraint (the algorithm is not stable)
    """

    def __init__(self, epsilon=0.1, gamma=1, stoperr=1e-6, numItermax=1000):
        super().__init__()
        self.epsilon = 0.1
        self.gamma = gamma
        self.stoperr = stoperr
        self.numItermax = numItermax
        self.w = None

    @torch.no_grad()
    def forward(self, P):
        # Q is the cost matrix with NxK
        P = P.clone().detach()
        P = -torch.log(torch.softmax(P / 0.1, dim=1))
        Q = torch.exp(- P / self.epsilon)

        # prior distribution
        Pa = torch.ones(Q.shape[0], 1).cuda() / Q.shape[0]  # how many samples
        Pb = torch.ones(Q.shape[1], 1).cuda() / Q.shape[1]  # how many prototypes
        b = torch.ones(Q.shape[1], 1).cuda() / Q.shape[1]
        # make the matrix sums to 1
        # sum_Q = torch.sum(Q)
        # Q /= sum_Q
        fi = self.gamma / (self.gamma + self.epsilon)
        err = 1
        last_b = b
        iternum = 0
        while err > self.stoperr and iternum < self.numItermax:
            a = Pa / (Q @ b)
            b = torch.pow(Pb / (Q.t() @ a), fi)
            err = torch.norm(b - last_b)
            last_b = b
            iternum += 1

        OT_plan = Q.shape[0] * a * Q * b.T
        # normalize
        # OT_plan /= OT_plan.sum(dim=1, keepdim=True)
        loss = torch.mean(torch.sum(OT_plan * P, dim=1))
        self.w = OT_plan.mean(dim=0, keepdim=True)
        reg = F.kl_div(torch.log(self.w + 1e-7), Pb.reshape(1, -1), reduction="batchmean")
        return OT_plan, loss, reg