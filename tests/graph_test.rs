#[cfg(test)]
mod tests {
    use mdarray::{view, View};
    use twinkgrad::graph::{Graph, Tensor};
    use twinkgrad::graph::NodeType::{DataNode, ComputeNode};

    #[test]
    fn graph_op() {
        // The node type should not matter here for the raw ops. The user will not be using this
        // interface directly.
        let g = &mut Graph::new();

        let x1 = g.add_val_l(3., "x1".to_string(), ComputeNode);
        let x2 = g.add_val_l(0.5, "x2".to_string(), DataNode);

        let w1 = g.add_val_l(-3.0, "w1".to_string(), ComputeNode);
        let w2 = g.add_val_l(1., "w2".to_string(), ComputeNode);

        let b = g.add_val_l(8., "b".to_string(), DataNode);

        let l1 = g.mul(x1, w1);
        let l2 = g.mul(x2, w2);

        let la = g.add(l1, l2, "la".to_string());

        let n = g.add(la, b, "n".to_string());

        let o = g.tanh(n);

        assert_eq!(g.get(n), -0.5);
        assert_eq!(g.get(o), -0.4620763851533083);
    }

    #[test]
    fn topo() {
        let g = &mut Graph::new();

        let x1 = g.add_val_l(2., "x1".to_string(), DataNode);
        let x2 = g.add_val_l(0.0, "x2".to_string(), DataNode);

        let w1 = g.add_val_l(-3., "w1".to_string(), DataNode);
        let w2 = g.add_val_l(1., "w2".to_string(), DataNode);

        let b = g.add_val_l(6.8813735870195432, "b".to_string(), DataNode);

        let l1 = g.mul(x1, w1);
        let l2 = g.mul(x2, w2);

        let la = g.add(l1, l2, "la".to_string());

        let n = g.add(la, b, "n".to_string());

        let o = g.tanh(n);

        g.set_grad(o, 1.);

        g.backward_retain_graph();

        assert_eq!(g.get_grad(n), 0.5000646207423279);
        assert_eq!(g.get_grad(la), 0.5000646207423279);
        assert_eq!(g.get_grad(b), 0.5000646207423279);
        assert_eq!(g.get_grad(l1), 0.5000646207423279);
        assert_eq!(g.get_grad(l2), 0.5000646207423279);
        assert_eq!(g.get_grad(x1), -1.500193862226984);
        assert_eq!(g.get_grad(w1), 1.0001292414846559);
        assert_eq!(g.get_grad(x2), 0.5000646207423279);
        assert_eq!(g.get_grad(w2), 0.);
    }

    #[test]
    fn neuron() {
        let g = &mut Graph::new();

        let neuron = &mut g.neuron_ones(3);

        let mut x = &mut vec!();
        for i in 0..3 {
            x.push(g.add_val_l(f64::from(i) as f64, format!("x{}", i.to_string()), DataNode));
        }

        let y = g.apply_neuron(neuron, x);

        // this is not tanh(4) because tanh is using a lower precision for e.
        assert_eq!(g.get(y), 0.9993287433665291);

        g.set_grad(y, 1.);

        g.backward_retain_graph();

        println!("{:?}", g.nodes.get(12));

        assert_eq!(g.nodes.get(12).unwrap().grad, 0.0013420626814738545);
    }

    #[test]
    fn layer() {
        let g = &mut Graph::new();

        let layer = &mut g.layer_ones(3, 4);

        let x = &vec!(
            g.add_val_l(2., "".to_string(), DataNode),
            g.add_val_l(3., "".to_string(), DataNode),
            g.add_val_l(-1., "".to_string(), DataNode),
        );

        let y = g.apply_layer(layer, x);

        println!("{:?}", y);

        assert_eq!(g.get(y[0]), 0.9999091100771555);
        assert_eq!(g.get(y[1]), 0.9999091100771555);
        assert_eq!(g.get(y[2]), 0.9999091100771555);
        assert_eq!(g.get(y[3]), 0.9999091100771555);
    }

    #[test]
    fn mlp_one() {
        let g = &mut Graph::new();

        let mlp = &mut g.mlp_ones(3, vec![4]);

        let x = &vec!(
            g.add_val_l(2., "".to_string(), DataNode),
            g.add_val_l(3., "".to_string(), DataNode),
            g.add_val_l(-1., "".to_string(), DataNode),
        );

        let y = g.apply_mlp(mlp, x);

        assert_eq!(g.get(y[0]), 0.9999091100771555);
        assert_eq!(g.get(y[1]), 0.9999091100771555);
        assert_eq!(g.get(y[2]), 0.9999091100771555);
        assert_eq!(g.get(y[3]), 0.9999091100771555);
    }

    #[test]
    fn mlp_two() {
        let g = &mut Graph::new();

        let mlp = &mut g.mlp_ones(3, vec![4, 3, 1]);

        let x = &vec!(
            g.add_val(2., DataNode), g.add_val(3., DataNode), g.add_val(-1., DataNode),
        );

        let y = g.apply_mlp(mlp, x);

        println!("{:?}", y);

        assert_eq!(g.get(y[0]), 0.9993283770985828);
    }

    #[test]
    fn loss() {
        let g = &mut Graph::new();

        let mlp = &mut g.mlp_ones(3, vec![4, 3, 1]);

        let xs = &vec!(
            g.tensor(vec!(2., 3., -1.)),
            g.tensor(vec!(3., -1., 0.5)),
            g.tensor(vec!(0.5, 1., 1.)),
            g.tensor(vec!(1., 1., -1.)),
        );

        let ys = g.tensor(vec!(1., -1., -1., 1.));

        let mut ypred = vec!();
        for i in 0..xs.len() {
            ypred.push(g.apply_mlp(mlp, &xs[i]));
        }

        assert_eq!(g.get(ypred[0][0]), 0.9993283770985828);
        assert_eq!(g.get(ypred[1][0]), 0.9993283719860206);
        assert_eq!(g.get(ypred[2][0]), 0.9993283719860206);
        assert_eq!(g.get(ypred[3][0]), 0.999328255237185);
    }

    #[test]
    fn loss_grad() {
        let g = &mut Graph::new();

        let mlp = &mut g.mlp_ones(3, vec![4, 1]);

        let xs = vec!(
            g.add_val_l(2., "x1".to_string(), DataNode),
            g.add_val_l(3., "x2".to_string(), DataNode),
            g.add_val_l(-1., "x3".to_string(), DataNode));

        let ys = g.add_val_l(1., "y1".to_string(), DataNode);

        let mut loss;

        for i in 0..2 {
            let ypred = g.apply_mlp(mlp, &xs);
            println!("ypred {:?}", g.get_node(ypred[0]));

            let a = g.sub(ys, ypred[0]);
            let b = g.add_val(2., DataNode);
            loss = g.pow(a, b);
            let k = g.get(ys) - g.get(ypred[0]);

            g.set_grad(loss, 1.);

            g.backward_retain_graph();

            let total = g.nodes.len();
            println!("loss {:?}", g.get(loss));
            if i == 1 {
                assert_eq!(g.get(loss), 0.);
            }

            for param in mlp.parameters() {
                let data = g.get(param);
                let new_data = data + -100000000. * g.get_grad(param);
                g.set_data(param, new_data);
            }
        }
    }

    #[test]
    fn tensor_index() {
        let mut x = Tensor::new(vec!(2, 4, 5));

        // 1st block, 3rd row, 4th col
        // 1 * (4*5) + 3 * (5) + 4 = 39
        x.mem_block[0] = 11.;
        x.mem_block[20] = 12.;
        x.mem_block[39] = 10.;

        // assert_eq!(x.get(vec!(0, 0, 0)), 11.);
        assert_eq!(x.get(vec!(1, 0, 0)), 12.);
        // assert_eq!(x.get(vec!(1, 3, 4)), 10.);
    }

    #[test]
    fn tensor_mul2d() {
        let mut x = &mut Tensor::new(vec!(3, 4));
        for i in 0..x.mem_block.len() {
            x.mem_block[i] = i as f64;
        }

        let mut y = &mut Tensor::new(vec!(4, 3));
        for i in 0..y.mem_block.len() {
            y.mem_block[i] = i as f64;
        }
        println!("{}", y.get(vec!(1, 1)));

        let g = &mut Graph::new();

        let z = g.matmul(x, y);

        assert_eq!(z.mem_block, vec!(42., 48., 54., 114., 136., 158., 186., 224., 262.));

        assert_eq!(z.get(vec!(0, 0)), 42.);
        assert_eq!(z.get(vec!(0, 1)), 48.);
        assert_eq!(z.get(vec!(0, 2)), 54.);
        assert_eq!(z.get(vec!(1, 0)), 114.);
        assert_eq!(z.get(vec!(1, 1)), 136.);
        assert_eq!(z.get(vec!(1, 2)), 158.);
        assert_eq!(z.get(vec!(2, 0)), 186.);
        assert_eq!(z.get(vec!(2, 1)), 224.);
        assert_eq!(z.get(vec!(2, 2)), 262.);
    }

    #[test]
    fn tensor_mul() {
        let mut x = &mut Tensor::new(vec!(2, 3, 2));
        for i in 0..x.mem_block.len() {
            x.mem_block[i] = i as f64;
        }

        let mut y = &mut Tensor::new(vec!(2, 2, 2));
        for i in 0..y.mem_block.len() {
            y.mem_block[i] = i as f64;
        }

        let g = &mut Graph::new();

        let z = g.matmul(x, y);

        assert_eq!(z.mem_block, vec!(
            2., 3., 6., 11., 10., 19.,
            66., 79., 86., 103., 106., 127.,
        ));

        assert_eq!(z.dims, vec!(2, 3, 2));

        assert_eq!(z.get(vec!(0, 0, 0)), 2.);
        assert_eq!(z.get(vec!(0, 0, 1)), 3.);
        assert_eq!(z.get(vec!(0, 1, 0)), 6.);
        assert_eq!(z.get(vec!(0, 1, 1)), 11.);
        assert_eq!(z.get(vec!(0, 2, 0)), 10.);
        assert_eq!(z.get(vec!(0, 2, 1)), 19.);

        assert_eq!(z.get(vec!(1, 0, 0)), 66.);
        assert_eq!(z.get(vec!(1, 0, 1)), 79.);
        assert_eq!(z.get(vec!(1, 1, 0)), 86.);
        assert_eq!(z.get(vec!(1, 1, 1)), 103.);
        assert_eq!(z.get(vec!(1, 2, 0)), 106.);
        assert_eq!(z.get(vec!(1, 2, 1)), 127.);
    }

    #[test]
    fn conv1d() {
        let g = &mut Graph::new();

        let x = g.tensor(vec!(1., 1., 2., 1.));

        let k = g.tensor(vec!(1., 0., -1.));

        let conv = g.apply_conv1d(x, k);

        assert_eq!(conv.len(), 2);
        assert_eq!(g.get(conv[0]), -1.);
        assert_eq!(g.get(conv[1]), 0.);
    }

    #[test]
    fn conv2d() {
        let g = &mut Graph::new();

        let x = view![
            [
                [1, 2],
                [2, 3]
            ],
            [
                [1, 2],
                [2, 3]
            ]
        ];

        let x = g.tensor(vec!(1., 1., 2., 1.));

        let k = g.tensor(vec!(1., 0., -1.));

        let conv = g.apply_conv1d(x, k);

        assert_eq!(conv.len(), 2);
        assert_eq!(g.get(conv[0]), -1.);
        assert_eq!(g.get(conv[1]), 0.);
    }
}
