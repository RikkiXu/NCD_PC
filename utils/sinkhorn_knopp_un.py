import torch
import torch.nn.functional as F
import torch.nn as nn


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, logits, queue=None):

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
        #self.sk.ini()

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
    def __init__(self, split, num_iters=3, epsilon=0.05, lr_w=0.01, num_outer_iters=10, gamma=5, num_unlabel=3, path=None, log=None):
        super().__init__()
        self.sk = SinkhornKnopp_im(num_iters=num_iters, epsilon=epsilon, split=split, num_unlabel=num_unlabel).cuda()
        self.opt = torch.optim.SGD(params=self.sk.parameters(), lr=lr_w, momentum=0.9)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.num_outer_iters = num_outer_iters
        self.gamma = gamma
        self.path = path
        self.lr_w = lr_w
        self.split = split


    def forward(self, features, head):

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
            self.gamma = 3*loss.detach() + 0.05
            total_loss = loss + self.gamma * reg
            self.opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sk.parameters(), 1)
            self.opt.step()

            if torch.norm(self.sk.K2 - last_w) < 1e-4:
                break
            last_w = self.sk.K2
        return Q, loss.item(), reg.item()