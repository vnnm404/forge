from graphxai.datasets import Benzene, Synth

bz = Benzene()

print(bz[0])

sy = Synth(num_samples=2000, shape1='cycle_6', shape2='house', seed=0)

print(sy[0])