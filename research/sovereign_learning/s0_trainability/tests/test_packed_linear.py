"""Synthetic-only PyTorch checks. No optimizer step or model training."""
import unittest

try:
    import torch
    from torch.nn import functional as F
except ImportError:
    torch=None

if torch is not None:
    from research.sovereign_learning.s0_trainability.packed_linear import (
        FrozenTernaryLinear, LoRAOverFrozen, normalized_fwht,
    )
    from research.sovereign_learning.s0_trainability.ptq1_codec import encode_ptq1_block


@unittest.skipIf(torch is None,"optional PyTorch unavailable")
class FrozenReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.q=torch.randint(-1,2,(3,256),dtype=torch.int8)
        self.s=torch.tensor([[0.5,1],[1,2],[0.25,0.75]],dtype=torch.float32)
        self.sign=torch.tensor([1 if i%3 else -1 for i in range(256)],dtype=torch.float64)
        self.x=torch.randn(4,256,dtype=torch.float64,requires_grad=True)

    def test_hadamard_involution_and_norm(self):
        y=normalized_fwht(self.x*self.sign,block_size=256)
        torch.testing.assert_close(normalized_fwht(y,block_size=256),self.x*self.sign,atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(y.square().sum(dim=1),self.x.square().sum(dim=1),atol=1e-11,rtol=1e-11)

    def test_frozen_forward_and_gradient_matches_dense(self):
        base=FrozenTernaryLinear(self.q,self.s,signs=self.sign,rotation_block=256)
        actual=base(self.x)
        dense=self.q.double()*self.s.double().repeat_interleave(128,dim=1)
        expected=F.linear(normalized_fwht(self.x*self.sign,block_size=256),dense)
        torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-11)
        self.assertEqual(list(base.parameters()),[])
        a=torch.autograd.grad(actual.square().sum(),self.x,retain_graph=True)[0]
        b=torch.autograd.grad(expected.square().sum(),self.x)[0]
        torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)

    def test_lora_gradient_uses_original_basis(self):
        base=FrozenTernaryLinear(self.q,self.s,signs=self.sign,rotation_block=256)
        model=LoRAOverFrozen(base,rank=2,alpha=4).double()
        with torch.no_grad():model.B.fill_(0.125)
        actual=model(self.x)
        dense=self.q.double()*self.s.double().repeat_interleave(128,dim=1)
        expected=F.linear(normalized_fwht(self.x*self.sign,block_size=256),dense)
        expected+=F.linear(F.linear(self.x,model.A),model.B)*2
        torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-11)
        actual.square().sum().backward()
        self.assertIsNone(base.trits.grad)
        self.assertIsNone(base.scales.grad)
        self.assertGreater(model.A.grad.abs().sum().item(),0)
        self.assertGreater(model.B.grad.abs().sum().item(),0)

    def test_payload_forward_matches_direct_reference(self):
        blob=b"".join(encode_ptq1_block(self.q[i,j*128:(j+1)*128].tolist(),self.s[i,j].item())
                      for i in range(3) for j in range(2))
        from_payload=FrozenTernaryLinear.from_payload(blob,rows=3,cols=256,fmt="ptq1")
        direct=FrozenTernaryLinear(self.q,self.s)
        torch.testing.assert_close(from_payload(self.x),direct(self.x),atol=1e-11,rtol=1e-11)

    def test_rejects_wrong_signs_or_transform_size(self):
        with self.assertRaises(ValueError):
            FrozenTernaryLinear(self.q,self.s,signs=self.sign[:-1],rotation_block=256)
        with self.assertRaises(ValueError):
            FrozenTernaryLinear(self.q,self.s,signs=self.sign)
        with self.assertRaises(ValueError):normalized_fwht(self.x,block_size=100)


if __name__ == "__main__":
    unittest.main()
