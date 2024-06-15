use std::iter::zip;
use crate::graph::Graph;

mod graph;
mod neuron;
mod layer;
mod mlp;

fn main() {
    let g = &mut Graph::new();

    let mlp = &mut g.mlp_ones(3, vec![4, 3, 1]);

    let xs = &vec!(
        vec!(g.add_val(2.), g.add_val(3.), g.add_val(-1.)),
        vec!(g.add_val(3.), g.add_val(-1.), g.add_val(0.5)),
        vec!(g.add_val(0.5), g.add_val(1.), g.add_val(1.)),
        vec!(g.add_val(1.), g.add_val(1.), g.add_val(-1.)),
    );

    let ys = vec!(
        g.add_val(1.), g.add_val(-1.), g.add_val(-1.), g.add_val(1.)
    );

    for _ in 0..42 {
        let mut ypred = vec!();
        for i in 0..xs.len() {
            ypred.push(g.apply_mlp(mlp, &xs[i]));
        }

        let ypreds = vec!(
            ypred[0][0], ypred[1][0], ypred[2][0], ypred[3][0]
        );

        let mut loss = g.add_val(0.);

        for (i, j) in zip(ypreds, ys.clone()) {
            let a = g.sub(i, j);
            let b = g.add_val(2.);
            let c = g.pow(a, b);
            loss = g.add(loss, c, "".to_string());
        }

        println!("loss: {:?}", g.get(loss));

        let params = mlp.parameters();
        for param in params {
            g.set_grad(param, 0.);
        }

        g.set_grad(loss, 1.);
        g.backward();

        let params = mlp.parameters();
        for param in params {
            let new_data = g.get(param) + -0.5 * g.get_grad(param);
            g.set_data(param, new_data);
        }

        println!("node count {}", g.nodes.len());
    }
}