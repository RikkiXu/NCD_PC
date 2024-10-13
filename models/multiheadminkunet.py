import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np
import MinkowskiEngine.MinkowskiFunctional as MF
from models.minkunet import MinkUNet34C


class EP(nn.Module):
    def __init__(self, output_dim, num_prototypes, D=3):
        super(EP, self).__init__()
        self.embedding = ME.MinkowskiConvolution(
            output_dim,
            output_dim // 2,
            kernel_size=1,
            bias=False,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)
        print("Equiangular Classifier")
        P = self.generate_random_orthogonal_matrix(output_dim // 2, num_prototypes)
        I = torch.eye(num_prototypes)
        one = torch.ones(num_prototypes, num_prototypes)
        M = np.sqrt(num_prototypes / (num_prototypes - 1)) * torch.matmul(P, I - ((1 / num_prototypes) * one))
        self.M = M.cuda()
        # self.magnitude = nn.Parameter(torch.randn(1, num_prototypes))

    def generate_random_orthogonal_matrix(self, feat_in, num_prototypes):
        # feat in has to be larger than num classes.
        a = np.random.random(size=(feat_in, num_prototypes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_prototypes), atol=1e-06), torch.max(
            torch.abs(torch.matmul(P.T, P) - torch.eye(num_prototypes)))
        return P

    def forward(self, x):
        x = self.embedding(x)
        x = self.relu(x)

        x = torch.nn.functional.normalize(x.F, dim=1, p=2)
        head = torch.nn.functional.normalize(self.M, dim=0, p=2)
        logits = x @ head
        # return logits*self.magnitude
        return logits


class LinearCls(nn.Module):
    def __init__(self, output_dim, num_prototypes, D=3):
        super().__init__()

        self.prototypes = ME.MinkowskiConvolution(
            output_dim,
            num_prototypes,
            kernel_size=1,
            bias=False,
            dimension=D)

    def forward(self, x):
        return self.prototypes(x).F


class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes, D=3):
        super().__init__()
        print("Cosine Classifier")
        self.prototypes = ME.MinkowskiConvolution(
            output_dim,
            num_prototypes,
            kernel_size=1,
            bias=False,
            dimension=D)

    def forward(self, x):
        x = torch.nn.functional.normalize(x.F, dim=1, p=2)
        head = torch.nn.functional.normalize(self.prototypes.kernel, dim=0, p=2)
        logits = x @ head
        return logits


class MultiHead(nn.Module):
    def __init__(
            self, input_dim, num_prototypes, num_heads
    ):
        super().__init__()
        self.num_heads = num_heads

        # prototypes
        self.prototypes = torch.nn.ModuleList(
            [Prototypes(input_dim, num_prototypes) for _ in range(num_heads)]
        )

        self.linears = torch.nn.ModuleList(
            [LinearCls(input_dim, num_prototypes) for _ in range(num_heads)]
        )

    def forward_head(self, head_idx, feats):
        return self.prototypes[head_idx](feats), self.linears[head_idx](feats), feats.F

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]


class MultiHeadMinkUnet(nn.Module):
    def __init__(
            self,
            num_labeled,
            num_unlabeled,
            overcluster_factor=None,
            num_heads=1
    ):
        super().__init__()

        # backbone -> pretrained model + identity as final
        self.encoder = MinkUNet34C(1, num_labeled)
        self.feat_dim = self.encoder.final.in_channels
        self.encoder.final = nn.Identity()

        self.head_lab = EP(output_dim=self.feat_dim, num_prototypes=num_labeled)
        if num_heads is not None:
            self.head_unlab = MultiHead(
                input_dim=self.feat_dim,
                num_prototypes=num_unlabeled,
                num_heads=num_heads
            )

        if overcluster_factor is not None:
            self.head_unlab_over = MultiHead(
                input_dim=self.feat_dim,
                num_prototypes=num_unlabeled * overcluster_factor,
                num_heads=num_heads
            )

    def forward_heads(self, feats):
        out = {"logits_lab": self.head_lab(feats)}
        if hasattr(self, "head_unlab"):
            logits_unlab, logits_unlab_linear, proj_feats_unlab = self.head_unlab(feats)
            out.update(
                {
                    "logits_unlab": logits_unlab,
                    "logits_unlab_linear": logits_unlab_linear,
                    "proj_feats_unlab": proj_feats_unlab,
                }
            )
        if hasattr(self, "head_unlab_over"):
            logits_unlab_over, proj_feats_unlab_over = self.head_unlab_over(feats)
            out.update(
                {
                    "logits_unlab_over": logits_unlab_over,
                    "proj_feats_unlab_over": proj_feats_unlab_over,
                }
            )
        return out

    def forward(self, views):
        if isinstance(views, list):
            feats = [self.encoder(view) for view in views]
            out = [self.forward_heads(f) for f in feats]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            feats = self.encoder(views)
            out = self.forward_heads(feats)
            out["feats"] = feats.F
            return out


class MinkUnet(nn.Module):
    def __init__(
            self,
            num_labeled,
            num_unlabeled,
            discover=True,
    ):
        super().__init__()

        # backbone -> pretrained model + identity as final
        self.encoder = MinkUNet34C(1, num_labeled)
        self.feat_dim = self.encoder.final.in_channels
        self.encoder.final = nn.Identity()

        self.head_lab = Prototypes(output_dim=self.feat_dim,
                                   num_prototypes=num_labeled)
        if discover:
            self.head_unlab = Prototypes(output_dim=self.feat_dim,
                                         num_prototypes=num_unlabeled)

            self.head_unlab_linears = EP(output_dim=self.feat_dim, num_prototypes=num_unlabeled)

    def forward_heads(self, feats):
        out = {"logits_lab": self.head_lab(feats)}
        if hasattr(self, "head_unlab"):
            logits_unlab = self.head_unlab(feats)
            logits_unlab_linear = self.head_unlab_linears(feats)
            proj_feats_unlab = feats.F
            out.update(
                {
                    "logits_unlab": logits_unlab,
                    "logits_unlab_linear": logits_unlab_linear,
                    "proj_feats_unlab": proj_feats_unlab,
                }
            )

        return out

    def forward(self, views):
        if isinstance(views, list):
            feats = [self.encoder(view) for view in views]
            out = [self.forward_heads(f) for f in feats]
            #import ipdb;ipdb.set_trace()
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            feats = self.encoder(views)
            #import ipdb;
            #ipdb.set_trace()
            out = self.forward_heads(feats)
            out["feats"] = feats.F
            return out