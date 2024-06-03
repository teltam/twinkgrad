use crate::graph::Graph;

mod graph;
mod neuron;
mod layer;
mod mlp;

fn main() {
    let g = &mut Graph::new();

    let n = g.neuron(10);

    let x = g.a_range(10);

    let res = g.apply_neuron(&n, &x);

    println!("----\n {:?} ----\n", res);

    let g = &mut Graph::new();
    let l = g.layer(2, 3);
    let x = &g.a_range(2);
    let res = g.apply_layer(&l, x);
    println!("{:?}", res);

    // let m = g.mlp(10, vec!(20, 25));
}