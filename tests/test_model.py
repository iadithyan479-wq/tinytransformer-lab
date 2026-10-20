import unittest
from tinytransformer.train import train, generate
class TestModel(unittest.TestCase):
    def test_training_and_generation(self):
        m,e,l=train('robot learns. '*20, steps=2, block_size=12); self.assertEqual(len(l),2); self.assertGreater(len(generate(m,e,'r',5)),1)
if __name__=='__main__': unittest.main()
