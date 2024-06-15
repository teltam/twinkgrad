#[cfg(test)]
mod tests {
    // use crate::graph::Graph;

    use twinkgrad::graph::Graph;

    #[test]
    fn graph_op() {
        let g = &mut Graph::new();

        let x1 = g.add_val_l(3., "x1".to_string());
        let x2 = g.add_val_l(0.5, "x2".to_string());

        let w1 = g.add_val_l(-3.0, "w1".to_string());
        let w2 = g.add_val_l(1., "w2".to_string());

        let b = g.add_val_l(8., "b".to_string());

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

        let x1 = g.add_val_l(2., "x1".to_string());
        let x2 = g.add_val_l(0.0, "x2".to_string());

        let w1 = g.add_val_l(-3., "w1".to_string());
        let w2 = g.add_val_l(1., "w2".to_string());

        let b = g.add_val_l(6.8813735870195432, "b".to_string());

        let l1 = g.mul(x1, w1);
        let l2 = g.mul(x2, w2);

        let la = g.add(l1, l2, "la".to_string());

        let n = g.add(la, b, "n".to_string());

        let o = g.tanh(n);

        g.set_grad(o, 1.);

        g.backward();

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
            x.push(g.add_val_l(f64::from(i) as f64, format!("x{}", i.to_string())));
        }

        let y = g.apply_neuron(neuron, x);

        // this is not tanh(4) because tanh is using a lower precision for e.
        assert_eq!(g.get(y), 0.9993287433665291);

        g.set_grad(y, 1.);

        g.backward();

        println!("{:?}", g.nodes.get(12));

        assert_eq!(g.nodes.get(12).unwrap().grad, 0.0013420626814738545);
    }

    #[test]
    fn layer() {
        let g = &mut Graph::new();

        let layer = &mut g.layer_ones(3, 4);

        let x = &vec!(
            g.add_val_l(2., "".to_string()),
            g.add_val_l(3., "".to_string()),
            g.add_val_l(-1., "".to_string()),
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
            g.add_val_l(2., "".to_string()),
            g.add_val_l(3., "".to_string()),
            g.add_val_l(-1., "".to_string()),
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
            g.add_val(2.), g.add_val(3.), g.add_val(-1.),
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
            vec!(g.add_val(2.), g.add_val(3.), g.add_val(-1.)),
            vec!(g.add_val(3.), g.add_val(-1.), g.add_val(0.5)),
            vec!(g.add_val(0.5), g.add_val(1.), g.add_val(1.)),
            vec!(g.add_val(1.), g.add_val(1.), g.add_val(-1.)),
        );

        let ys = vec!(
            g.add_val(1.), g.add_val(-1.), g.add_val(-1.), g.add_val(1.)
        );

        let mut ypred = vec!();
        for i in 0..xs.len() {
            ypred.push(g.apply_mlp(mlp, &xs[i]));
        }

        // for i in 0..xs.len() {
        //     assert_eq!(ypred[i][2].len(), 1);
        // }

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
            g.add_val_l(2., "x1".to_string()),
            g.add_val_l(3., "x2".to_string()),
            g.add_val_l(-1., "x3".to_string()));

        let ys = g.add_val_l(1., "y1".to_string());

        let mut loss;

        for i in 0..2 {
            let ypred = g.apply_mlp(mlp, &xs);
            println!("ypred {:?}", g.get_node(ypred[0]));

            let a = g.sub(ys, ypred[0]);
            let b = g.add_val(2.);
            loss = g.pow(a, b);
            let k = g.get(ys) - g.get(ypred[0]);

            g.set_grad(loss, 1.);

            g.backward();

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
}
