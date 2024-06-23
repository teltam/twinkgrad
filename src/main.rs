use std::iter::zip;
use crate::graph::Graph;
use crate::graph::NodeType::{ComputeNode, DataNode};

mod graph;
mod neuron;
mod layer;
mod mlp;

fn main() {
    let g = &mut Graph::new();

    let mlp = &mut g.mlp_ones(3, vec![4, 3, 1]);

    let xs = &vec!(
        g.tensor(vec!(2., 3., -1.)),
        g.tensor(vec!(3., -1., 0.5)),
        g.tensor(vec!(0.5, 1., 1.)),
        g.tensor(vec!(1., 1., -1.)),
    );

    let ys = g.tensor(vec!(1., -1., -1., 1.));

    for _ in 0..42 {
        let mut ypred = vec!();
        for i in 0..xs.len() {
            ypred.push(g.apply_mlp(mlp, &xs[i]));
        }

        let ypreds = vec!(
            ypred[0][0], ypred[1][0], ypred[2][0], ypred[3][0]
        );

        let mut loss = g.add_val(0., ComputeNode);

        for (i, j) in zip(ypreds, ys.clone()) {
            let a = g.sub(i, j);
            let b = g.add_val(2., ComputeNode);
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