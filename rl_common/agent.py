import random, numpy as np, collections, tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class DQNAgent:
    def __init__(self, state_dim, n_actions, gamma=0.99, lr=1e-3,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.995,
                 buf_size=200_000, batch_size=256):
        self.state_dim=state_dim; self.n_actions=n_actions
        self.gamma=gamma; self.epsilon=eps_start
        self.eps_end=eps_end; self.eps_decay=eps_decay
        self.batch_size=batch_size
        self.memory=collections.deque(maxlen=buf_size)
        self.model=self._build(lr); self.target=self._build(lr)
        self.update_target()

    def _build(self, lr):
        m=models.Sequential([
            layers.Input((self.state_dim,)),
            layers.Dense(128,activation="relu"),
            layers.Dense(128,activation="relu"),
            layers.Dense(64,activation="relu"),
            layers.Dense(self.n_actions,activation="linear"),
        ])
        m.compile(optimizer=optimizers.Adam(learning_rate=lr),loss="mse"); return m

    def update_target(self): self.target.set_weights(self.model.get_weights())
    def act(self,s):
        if random.random()<self.epsilon: return random.randrange(self.n_actions)
        q=self.model.predict(s[None,:],verbose=0)[0]; return int(np.argmax(q))
    def remember(self,s,a,r,s2,d): self.memory.append((s.astype(np.float32),a,r,s2.astype(np.float32),d))
    def replay(self):
        if len(self.memory)<self.batch_size: return 0.0
        batch=random.sample(self.memory,self.batch_size)
        s=np.array([b[0] for b in batch],dtype=np.float32)
        a=np.array([b[1] for b in batch],dtype=np.int32)
        r=np.array([b[2] for b in batch],dtype=np.float32)
        s2=np.array([b[3] for b in batch],dtype=np.float32)
        d=np.array([b[4] for b in batch],dtype=np.float32)
        q=self.model.predict(s,verbose=0)
        qt=self.target.predict(s2,verbose=0)
        maxn=np.max(qt,axis=1)
        tq=q.copy()
        tq[np.arange(self.batch_size),a]=r+(1-d)*self.gamma*maxn
        hist=self.model.fit(s,tq,epochs=1,verbose=0,batch_size=self.batch_size)
        self.epsilon=max(self.eps_end,self.epsilon*self.eps_decay)
        return float(hist.history["loss"][0])
    def save(self,p): self.model.save(p)
    def load(self,p):
        # weights-only loader
        import numpy as np
        _ = self.model(np.zeros((1, self.state_dim), dtype=np.float32), training=False)
        self.model.load_weights(p)
        self.update_target()
