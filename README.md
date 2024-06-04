# twinkgrad
Inspired by Karpathy's micrograd, twinkgrad is micrograd in Rust. The interface is a little different from the python version
given how python objects and dunder methods work. The main actor is the compute graph itself and all operation go through it.

![](./spy.jpeg)

## Example

1. Setup the compute graph,
```rust
    let g = &mut Graph::new();

    let mlp = &mut g.mlp_ones(3, vec![4, 3, 1]);

    let xs = &vec!(
        vec!(g.add_val(2.), g.add_val(3.), g.add_val(-1.)),
        vec!(g.add_val(3.), g.add_val(-1.), g.add_val(0.5)),
        vec!(g.add_val(0.5), g.add_val(1.), g.add_val(1.)),
        vec!(g.add_val(1.), g.add_val(1.), g.add_val(-1.)),
    );
```

2. Setup the predictions and infer,
```rust
        let ys = vec!(
            g.add_val(1.), g.add_val(-1.), g.add_val(-1.), g.add_val(1.)
        );

        let mut ypred = vec!();
        for i in 0..xs.len() {
            ypred.push(g.apply_mlp(mlp, &xs[i]));
        }

        let ypreds = vec!(
            ypred[0][0], ypred[1][0], ypred[2][0], ypred[3][0]
        );
```

3. Compute loss,
```rust
        let mut loss = g.add_val(0.);

        for (i, j) in zip(ypreds, ys) {
            let a = g.sub(i, j);
            let b = g.add_val(2.);
            let c = g.pow(a, b);
            loss = g.add(loss, c, "".to_string());
        }

        println!("loss: {:?}", g.get(loss));
```

4. Perform grad,
```rust
        g.set_grad(loss, 1.);
        g.backward();
```

5. Update params,
```rust
        let params = mlp.parameters();
        for param in params {
            let new_data = g.get(param) + -1. * g.get_grad(param);
            g.set_data(param, new_data);
        }
```
