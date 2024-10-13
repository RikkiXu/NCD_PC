import torch
import torch.nn.functional as F
import torch.nn as nn

class SinkhornKnopp(torch.nn.Module):
    def __init__(self, split=0, num_iters=3, epsilon=0.05, num_unlabel=3):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        if split == 3:
            self.ww = torch.zeros((4))
            self.ww = F.softmax(self.ww, dim=0).cuda()
        else:
            self.ww = torch.zeros((5))
            self.ww = F.softmax(self.ww, dim=0).cuda()

    def update(self, targets):
        psuedo_labels = torch.argmax(targets, dim=-1).reshape(-1)
        nums = torch.unique(psuedo_labels, return_counts=True)[1]
        nums = nums / sum(nums)
        nums = torch.sort(nums, descending=True)[0]
        self.ww = self.ww.cuda()
        self.ww = 0.01 * nums + self.ww * 0.99


    @torch.no_grad()
    def forward(self, features, head, split=0, mode='default', queue=None, return_logit=False):

        features = torch.nn.functional.normalize(features, dim=1, p=2)
        head = torch.nn.functional.normalize(head, dim=1, p=2)

        logits = features@head

        logits = logits.to(torch.float64)
        Q = torch.exp(logits / self.epsilon).t()
        B = Q.shape[1]


        sum_Q = torch.sum(Q)
        Q /= sum_Q


        for it in range(self.num_iters):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q = (Q.t() * self.ww).t()
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        to_ret = Q.t() if queue is None else Q.t()[:-queue.shape[0]]
        self.update(to_ret)
        if return_logit:
            return to_ret, logits
        else:
            return to_ret

class SinkhornKnopp_epoch(torch.nn.Module):
    def __init__(self, split=0, num_iters=3, epsilon=0.05, num_unlabel=3, ema=0.8):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.split = split
        if split == 3:
            self.ww = torch.zeros((4))
            self.ww = F.softmax(self.ww, dim=0).cuda()
            self.nums = torch.zeros((4))
        else:
            self.ww = torch.zeros((5))
            self.ww = F.softmax(self.ww, dim=0).cuda()
            self.nums = torch.zeros((5))
        self.count = 0
        self.ema = ema


    def update(self, targets):
        psuedo_labels = torch.argmax(targets, dim=-1).reshape(-1)
        self.nums = torch.unique(psuedo_labels, return_counts=True)[1] + self.nums.cuda()



    def updatew(self):
        nums = self.nums / sum(self.nums)
        nums = torch.sort(nums, descending=True)[0]
        self.ww = self.ww.cuda()
        self.ww = (1-self.ema) * nums + self.ww * self.ema
        if self.split == 3:
            self.nums = torch.zeros((4))
        else:
            self.nums = torch.zeros((5))


    @torch.no_grad()
    def forward(self, features, head, split=0, mode='default', queue=None, return_logit=False):

        features = torch.nn.functional.normalize(features, dim=1, p=2)
        head = torch.nn.functional.normalize(head, dim=1, p=2)

        logits = features@head

        logits = logits.to(torch.float64)
        Q = torch.exp(logits / self.epsilon).t()
        B = Q.shape[1]

        sum_Q = torch.sum(Q)
        Q /= sum_Q


        for it in range(self.num_iters):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q = (Q.t() * self.ww).t()
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        to_ret = Q.t() if queue is None else Q.t()[:-queue.shape[0]]
        self.update(to_ret)
        if self.count % 1400 == 0:
            self.updatew()
        self.count = self.count + 1
        if return_logit:
            return to_ret, logits
        else:
            return to_ret



class SinkhornKnopp_gt(torch.nn.Module):
    def __init__(self, split, num_iters=3, epsilon=0.05, lr_w=0.01, num_outer_iters=10, gamma=5, num_unlabel=3, path=None):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.w = torch.nn.Parameter(torch.ones(1, num_unlabel) / num_unlabel)
        self.w0 = torch.ones(1, num_unlabel).cuda() / num_unlabel
        if split == 3:
            self.ww = torch.zeros((4))
            self.ww = F.softmax(self.ww, dim=0).cuda()
        else:
            self.ww = torch.zeros((5))
            self.ww = F.softmax(self.ww, dim=0).cuda()
        self.split = split

    def get_w(self):
        return torch.log_softmax(self.w, dim=1)

    def get_w0(self):
        return torch.softmax(self.w0, dim=1)

    @torch.no_grad()
    def forward(self, features, head, unseen_num):

        features = torch.nn.functional.normalize(features, dim=1, p=2)
        head = torch.nn.functional.normalize(head, dim=1, p=2)

        logits = features@head

        logits = logits.to(torch.float64)
        Q = torch.exp(logits / self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for i in range(len(unseen_num)):
            self.ww[i] = unseen_num[i] / sum(unseen_num)
        self.ww = self.ww.cuda()



        for it in range(self.num_iters):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q = (Q.t() * self.ww).t()
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B
        to_ret = Q.t()

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


    def forward(self, features, head, split=0, mode='default', queue=None, return_logit=False):
        features = features.detach()
        head = head.detach()
        features = torch.nn.functional.normalize(features, dim=1, p=2)
        head = torch.nn.functional.normalize(head, dim=1, p=2)
        logits = features@head
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
    def __init__(self, split, num_iters=3, epsilon=0.05, lr_w=0.01, num_outer_iters=10, gamma=5, num_unlabel=3, path=None):
        super().__init__()
        self.sk = SinkhornKnopp_im(num_iters=num_iters, epsilon=epsilon, split=split, num_unlabel=num_unlabel).cuda()
        self.opt = torch.optim.SGD(params=self.sk.parameters(), lr=lr_w, momentum=0.9)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.num_outer_iters = num_outer_iters
        self.gamma = gamma
        self.count = 0
        self.path = path
        self.lr_w = lr_w
        self.split =split


    def forward(self, features, head, unseen_num):

        features = features.detach()
        self.count = self.count + 1
        last_w = torch.zeros_like(self.sk.K2)
        if self.count % 2000 == 0 :
            save_w0 = []
            save_w1 = []
            save_w2 = []
            save_w3 = []

            save_w_1 = []
            loss_1 = []
            loss_2 = []
            for it in range(self.num_outer_iters):
                Q, preds = self.sk(features, head, return_logit=True)
                prob = F.softmax(preds / 0.1, dim=1)
                loss = - torch.mean(torch.sum(Q * torch.log(prob), dim=1))
                input = self.sk.get_w().t().cuda()
                target = self.sk.get_w0().t().cuda()
                reg = self.kl_loss(input, target)
                total_loss = loss + self.gamma * reg
                self.opt.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sk.parameters(), 1)
                self.opt.step()
                save_w0.append(self.sk.K2[0][0].item())
                save_w1.append(self.sk.K2[0][1].item())
                save_w2.append(self.sk.K2[0][2].item())
                save_w3.append(self.sk.K2[0][3].item())
                save_w_1.append(self.sk.K2[0][-1].item())
                loss_1.append(loss.item())
                loss_2.append(self.gamma * reg.item())
                if torch.norm(self.sk.K2 - last_w) < 1e-4:
                    break
                last_w = self.sk.K2

            import matplotlib.pyplot as plt
            #if self.split == 3:
            #   ss = [save_w0[-1], save_w1[-1], save_w2[-1], save_w3[-1]]
            #else:
            #    ss = [save_w0[-1], save_w1[-1], save_w2[-1], save_w3[-1], save_w_1[-1]]
            #ss.sort()

            #print(unseen_num)
            #unseen_num = sorted(unseen_num)
            #sum_un = sum(unseen_num).item()
            #for i in range(len(unseen_num)):
            #    unseen_num[i] = round((unseen_num[i] / sum_un - ss[i]).item()*100, 2)

            x = list(range(len(save_w0)))
            plt.subplot(3, 3, 1)
            plt.plot(x, save_w0)
            plt.title("w0")

            plt.subplot(3, 3, 2)
            plt.plot(x, save_w1)
            plt.title("w1")

            plt.subplot(3, 3, 3)
            plt.plot(x, save_w2)
            plt.title("w2")

            plt.subplot(3, 3, 4)
            plt.plot(x, save_w3)
            plt.title("w3")


            plt.subplot(3, 3, 5)
            plt.plot(x, save_w_1)
            plt.title("w4")

            plt.subplot(3, 3, 6)
            plt.plot(x, loss_1)
            plt.title("Lu")

            plt.subplot(3, 3, 7)
            plt.plot(x, loss_2)
            plt.title("KL")

            #plt.suptitle(unseen_num)
            plt.savefig(self.path+'/w'+'_'+str(self.count)+'_'+str(self.lr_w)+'sgd.png')
            plt.close()



        else:
            for it in range(self.num_outer_iters):
                Q, preds = self.sk(features, head, return_logit=True)
                prob = F.softmax(preds / 0.1, dim=1)
                loss = - torch.mean(torch.sum(Q * torch.log(prob), dim=1))
                input = self.sk.get_w().t().cuda()
                target = self.sk.get_w0().t().cuda()
                reg = self.kl_loss(input, target)
                total_loss = loss + self.gamma * reg
                self.opt.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sk.parameters(), 1)
                self.opt.step()

                if torch.norm(self.sk.K2 - last_w) < 1e-4:
                    break
                last_w = self.sk.K2



        return Q