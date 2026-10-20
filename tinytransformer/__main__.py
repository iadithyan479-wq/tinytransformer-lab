import argparse
from .train import train, generate
p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=300); p.add_argument('--prompt',default='AI '); a=p.parse_args()
text=('AI and robotics learn from data. Small models make big ideas visible. ' * 30)
model, enc, losses=train(text,a.steps); print('final_loss:',round(losses[-1],4)); print(generate(model,enc,a.prompt))
