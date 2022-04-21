Give basis functions $B_i(x)$ and a function $Y(x)$, find $w_i$ so $Y(x) \sim \tanh(\sum_i w_i B_i(x))$. $x$ can be scalar or vector.

Set $Y(x)$ to either 1 or -1, although numbers in between *potentially* increase stability.

Can be used to do binary classification. The motivation of this is to find "better" fits to binary images and implicitly defined 3D models.
