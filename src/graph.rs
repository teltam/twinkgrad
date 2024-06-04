use std::borrow::Cow;
use std::iter::zip;
use rand::{Rng, thread_rng};
use crate::layer::Layer;
use crate::mlp::MLP;
use crate::neuron::Neuron;

#[derive(Clone)]
pub struct Graph<T> {
    pub nodes: Vec<Node<T>>,
}

#[derive(Clone, Debug)]
pub struct Node<T> {
    pub label: Cow<'static, str>,
    pub data: T,

    pub grad: T,

    pub prev: Vec<NodeRef>,

    pub op: Ops,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeRef {
    pub node_id: usize,
}

#[derive(Clone, Debug)]
pub enum Ops {
    TANH,
    EXP,
    DIV,
    MUL,
    ADD,
    SUB,
    LEAF,
    EMPTY,
}

impl Node<f64> {
    pub fn new() -> Self {
        Node {
            data: 0.,
            grad: 0.,
            prev: vec![],
            op: Ops::EMPTY,

            label: Cow::from(""),
        }
    }

    pub fn new_v(val: f64, label: String) -> Self {
        let mut n: Node<f64> = Node::new();
        // if (val.is_nan()) {
        //     panic!();
        // }
        n.data = val;
        n.op = Ops::LEAF;
        n.label = Cow::from(label);
        n
    }

    pub fn new_op(val: f64, _prev: Vec<usize>, prev: Vec<NodeRef>, op: Ops) -> Self {
        let mut v: Node<f64> = Node::new();
        // if (val.is_nan()) {
        //     panic!();
        // }
        v.data = val;
        v.prev = prev;
        v.op = op;
        v
    }

    pub fn new_op_l(val: f64, _prev: Vec<usize>, prev: Vec<NodeRef>, op: Ops, label: String) -> Self {
        // if (val.is_nan()) {
        //     panic!();
        // }
        let mut v: Node<f64> = Node::new();
        v.data = val;
        v.prev = prev;
        v.op = op;
        v.label = Cow::from(label);
        v
    }
}

impl Graph<f64> {
    pub fn new() -> Self {
        Graph { nodes: vec![] }
    }
}

impl Graph<f64> {
    pub fn add_val_l(&mut self, val: f64, label: String) -> NodeRef {
        self.nodes.push(Node::new_v(val, label));

        NodeRef {
            node_id: self.nodes.len() - 1,
        }
    }

    pub fn add_val(&mut self, val: f64) -> NodeRef {
        self.nodes.push(Node::new_v(val, "".to_string()));

        NodeRef {
            node_id: self.nodes.len() - 1,
        }
    }

    pub fn a_range(&mut self, l: u8) -> Vec<NodeRef> {
        let mut res = vec!();
        for i in 0..l {
            res.push(self.add_val_l(f64::from(i), "".to_string()));
        }

        res
    }

    pub fn add_val_prev(
        &mut self,
        val: f64,
        _prev: Vec<usize>,
        prev: Vec<NodeRef>,
        op: Ops,
    ) -> NodeRef {
        self.nodes.push(Node::new_op(val, _prev, prev, op));

        NodeRef {
            node_id: self.nodes.len() - 1,
        }
    }

    pub fn add_val_prev_l(
        &mut self,
        val: f64,
        _prev: Vec<usize>,
        prev: Vec<NodeRef>,
        op: Ops,
        label: String,
    ) -> NodeRef {
        self.nodes.push(Node::new_op_l(val, _prev, prev, op, label));

        NodeRef {
            node_id: self.nodes.len() - 1,
        }
    }

    pub fn neuron(&mut self, l: usize) -> Neuron {
        let mut rng = thread_rng();

        let mut ws = vec!();
        for i in 0..l {
            ws.push(self.add_val_l(rng.gen::<f64>(), format!("w_{}", i)));
        }

        let b = self.add_val_l(rng.gen::<f64>(), "b".to_string());

        Neuron { ws, b, }
    }

    pub fn neuron_ones(&mut self, l: usize) -> Neuron {
        let mut ws = vec!();
        for i in 0..l {
            ws.push(self.add_val_l(1., format!("w_{}", i)));
        }

        let b = self.add_val_l(1., "b".to_string());

        Neuron { ws, b, }
    }

    pub fn layer(&mut self, lin: usize, out:usize) -> Layer {
        let mut ls = vec!();
        for _ in 0..out {
            ls.push(self.neuron(lin));
        }

        Layer { ls }
    }

    pub fn layer_ones(&mut self, lin: usize, out:usize) -> Layer {
        let mut ls = vec!();
        for _ in 0..out {
            ls.push(self.neuron_ones(lin));
        }

        Layer { ls }
    }

    pub fn mlp(&mut self, lin: usize, lout: Vec<usize>) -> MLP {
        let mut dims = vec!(lin);
        dims.extend(lout);

        let mut res = vec!();

        let len = dims.len();
        let a = dims.clone();
        let b = dims.clone();
        for (i, j) in zip(&a[0..len-1], &b[1..len]) {
            res.push(self.layer(*i, *j));
        }

        MLP { ls: res }
    }

    pub fn mlp_ones(&mut self, lin: usize, lout: Vec<usize>) -> MLP {
        let mut dims = vec!(lin);
        dims.extend(lout);

        let mut res = vec!();

        let len = dims.len();
        let a = dims.clone();
        let b = dims.clone();
        for (i, j) in zip(&a[0..len-1], &b[1..len]) {
            res.push(self.layer_ones(*i, *j));
        }

        MLP { ls: res }
    }

    pub fn get(&mut self, x: NodeRef) -> f64 {
        let node_id = x.node_id;

        return self.nodes.get(node_id).unwrap().data;
    }

    pub fn get_node(&mut self, x: NodeRef) -> Node<f64> {
        let node_id = x.node_id;

        return self.nodes.get(node_id).unwrap().clone();
    }

    pub fn get_node_vec(&mut self, x: Vec<NodeRef>) -> Vec<Node<f64>> {
        let mut res = vec!();

        for _x in x.iter() {
            res.push(self.get_node(_x.clone()))
        }

        return res;
    }

    pub fn get_grad(&mut self, x: NodeRef) -> f64 {
        let node_id = x.node_id;

        return self.nodes.get(node_id).unwrap().grad;
    }

    pub fn set_grad(&mut self, x: NodeRef, grad: f64) {
        let node_id = x.node_id;

        self.nodes.get_mut(node_id).unwrap().grad = grad;
    }

    pub fn set_data(&mut self, x: NodeRef, data: f64) {
        let node_id = x.node_id;

        self.nodes.get_mut(node_id).unwrap().data = data;
    }

    pub fn add(&mut self, x1: NodeRef, x2: NodeRef, label: String) -> NodeRef {
        let a1 = self.nodes.get_mut(x1.node_id).unwrap().data;
        let a2 = self.nodes.get_mut(x2.node_id).unwrap().data;

        self.add_val_prev_l(
            a1 + a2,
            vec![x1.node_id, x2.node_id],
            vec![x1, x2],
            Ops::ADD,
            label,
        );

        NodeRef {
            node_id: self.nodes.len() - 1,
        }
    }

    pub fn sub(&mut self, x1: NodeRef, x2: NodeRef) -> NodeRef {
        let a1 = self.nodes.get_mut(x1.node_id).unwrap().data;
        let a2 = self.nodes.get_mut(x2.node_id).unwrap().data;

        self.add_val_prev(
            a1 - a2,
            vec![x1.node_id, x2.node_id],
            vec![x1, x2],
            Ops::SUB,
        );

        NodeRef {
            node_id: self.nodes.len() - 1,
        }
    }

    pub fn mul(&mut self, x1: NodeRef, x2: NodeRef) -> NodeRef {
        let a1 = self.nodes.get_mut(x1.node_id).unwrap().data;
        let a2 = self.nodes.get_mut(x2.node_id).unwrap().data;

        self.add_val_prev(
            a1 * a2,
            vec![x1.node_id, x2.node_id],
            vec![x1, x2],
            Ops::MUL,
        );

        NodeRef {
            node_id: self.nodes.len() - 1,
        }
    }

    pub fn div(&mut self, x1: NodeRef, x2: NodeRef) -> NodeRef {
        let a1 = self.nodes.get_mut(x1.node_id).unwrap().data;
        let a2 = self.nodes.get_mut(x2.node_id).unwrap().data;

        self.add_val_prev(
            a1 / a2,
            vec![x1.node_id, x2.node_id],
            vec![x1, x2],
            Ops::DIV,
        );

        NodeRef {
            node_id: self.nodes.len() - 1,
        }
    }

    pub fn pow(&mut self, x1: NodeRef, x2: NodeRef) -> NodeRef {
        let a1 = self.nodes.get_mut(x1.node_id).unwrap().data;
        let a2 = self.nodes.get_mut(x2.node_id).unwrap().data;

        self.add_val_prev(
            f64::powf(a1, a2),
            vec![x1.node_id, x2.node_id],
            vec![x1, x2],
            Ops::EXP,
        );

        NodeRef {
            node_id: self.nodes.len() - 1,
        }
    }

    pub fn tanh(&mut self, x1: NodeRef) -> NodeRef {
        let a1 = self.nodes.get_mut(x1.node_id).unwrap().data;

        let t = (f64::powf(2.718, 2. * a1) - 1.) / (f64::powf(2.718, 2. * a1) + 1.);

        self.add_val_prev(t, vec![x1.node_id], vec![x1], Ops::TANH);

        // if t.is_nan() {
        //     println!("---- in tanh {:?}, {:?}, {}, {}", self.get_node(x1), t, (f64::powf(2.718, 2. * a1) - 1.), (f64::powf(2.718, 2. * a1) + 1.));
        // }

        NodeRef {
            node_id: self.nodes.len() - 1,
        }
    }

    pub fn apply_neuron(&mut self, n: &Neuron, x: &Vec<NodeRef>) -> NodeRef {
        // TODO assert lens of x and w.
        // TODO do we need to clean up a graph if we don't use?

        let l = n.ws.len();
        let mut res = n.b;
        for i in 0..l {
            let w_x_i = self.mul(n.ws[i], x[i]);
            res = self.add(res, w_x_i, format!("w_x_{}", i));
            // if self.get(res).is_nan() {
            //     println!("--- NAN applying neuron w, id, x id {:?}, {:?} {:?} {:?}", self.get(n.ws[i]), n.ws[i].node_id, self.get(x[i]), x[i].node_id);
            // }
        }


        return self.tanh(res);
    }

    pub fn apply_layer(&mut self, layer: &Layer, x: &Vec<NodeRef>) -> Vec<NodeRef> {
        let l = layer.ls.len();

        let mut res = vec!();

        for i in 0..l {
            let neuron= &layer.ls[i];

            // let nodes = self.get_node_vec(x.clone());
            // for node in nodes.iter() {
            //     println!("----- apply layer would have nans {:?}", node);
            // }

            let act = self.apply_neuron(neuron, x);
            res.push(act);
        }

        return res;
    }

    pub fn apply_mlp(&mut self, mlp: &MLP, x: &Vec<NodeRef>) -> Vec<Vec<NodeRef>> {
        // TODO assert sizes are the same for first layer.
        let l = mlp.ls.len();

        let mut res = vec!();

        res.push(x.clone());

        for i in 0..l {
            let layer = &mlp.ls[i];
            let last = res.last().unwrap();
            // println!("----- last {:?} ", self.get_node_vec(last.clone()));
            let act = self.apply_layer(layer, last);
            res.push(act);
        }

        return res[1..res.len()].to_vec();
    }

    // ==== Backward methods ====

    pub fn backward(&mut self) {
        let node_list = self.nodes.clone();

        // println!("----");
        // for (i, node) in node_list.iter().enumerate() {
        //     println!("{}, {:?}", i, node);
        // }
        // println!("----");

        // already topo sorted
        for (i, node) in node_list.iter().rev().enumerate() {
            let ii = node_list.len() - 1 - i;

            match &node.op {
                Ops::TANH => self.tanh_backward(ii),
                Ops::EXP => self.pow_backward(ii),
                Ops::DIV => self.div_backward(ii),
                Ops::MUL => self.mul_backward(ii),
                Ops::ADD => self.add_backward(ii),
                Ops::SUB => self.sub_backward(ii),
                Ops::LEAF => continue,
                Ops::EMPTY => {
                    panic!("unsupported op node_label: {}, op: {:?}", node.label, node.op)
                }
            }
        }
    }

    fn add_backward(&mut self, i: usize) {
        let node = self.nodes.get(i).unwrap().clone();

        let c1 = node.prev.get(0).unwrap();
        let c2 = node.prev.get(1).unwrap();

        let mut c1_node = self.nodes.get_mut(c1.node_id).unwrap().clone();
        let mut c2_node = self.nodes.get_mut(c2.node_id).unwrap().clone();

        c1_node.grad += 1. * node.grad;
        c2_node.grad += 1. * node.grad;

        self.nodes[c1.node_id] = c1_node;
        self.nodes[c2.node_id] = c2_node;
    }

    fn sub_backward(&mut self, i: usize) {
        let node = self.nodes.get(i).unwrap().clone();

        let c1 = node.prev.get(0).unwrap();
        let c2 = node.prev.get(1).unwrap();

        let mut c1_node = self.nodes.get_mut(c1.node_id).unwrap().clone();
        let mut c2_node = self.nodes.get_mut(c2.node_id).unwrap().clone();

        c1_node.grad += 1. * node.grad;
        c2_node.grad += -1. * node.grad;

        self.nodes[c1.node_id] = c1_node;
        self.nodes[c2.node_id] = c2_node;
    }

    fn mul_backward(&mut self, i: usize) {
        let node = self.nodes.get(i).unwrap().clone();

        let c1 = node.prev.get(0).unwrap();
        let c2 = node.prev.get(1).unwrap();

        let mut c1_node = self.nodes.get_mut(c1.node_id).unwrap().clone();
        let mut c2_node = self.nodes.get_mut(c2.node_id).unwrap().clone();

        c1_node.grad += c2_node.data * node.grad;
        c2_node.grad += c1_node.data * node.grad;

        self.nodes[c1.node_id] = c1_node;
        self.nodes[c2.node_id] = c2_node;
    }

    fn div_backward(&mut self, i: usize) {
        let node = self.nodes.get(i).unwrap().clone();

        let c1 = node.prev.get(0).unwrap();
        let c2 = node.prev.get(1).unwrap();

        let mut c1_node = self.nodes.get_mut(c1.node_id).unwrap().clone();
        let mut c2_node = self.nodes.get_mut(c2.node_id).unwrap().clone();

        c1_node.grad += (1. / c2_node.data) * node.grad;
        c2_node.grad += (-c1_node.data / f64::powf(c2_node.data, 2.)) * node.grad;

        self.nodes[c1.node_id] = c1_node;
        self.nodes[c2.node_id] = c2_node;
    }

    fn tanh_backward(&mut self, i: usize) {
        let node = self.nodes.get(i).unwrap().clone();

        let c1 = node.prev.get(0).unwrap();

        let mut c1_node = self.nodes.get_mut(c1.node_id).unwrap().clone();

        let g = (1. - f64::powf(node.data, 2.)) * node.grad;

        c1_node.grad += g;

        self.nodes[c1.node_id] = c1_node;

        // println!("tanh grad: {:?} {:?}", self.nodes[c1.node_id], g);
    }

    fn pow_backward(&mut self, i: usize) {
        let node = self.nodes.get(i).unwrap().clone();

        let c1 = node.prev.get(0).unwrap();
        let c2 = node.prev.get(1).unwrap();

        let mut c1_node = self.nodes.get_mut(c1.node_id).unwrap().clone();
        let c2_node = self.nodes.get_mut(c2.node_id).unwrap().clone();

        let g = c2_node.data * f64::powf(c1_node.data, c2_node.data - 1.) * node.grad;

        c1_node.grad += g;

        self.nodes[c1.node_id] = c1_node;
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;
    use crate::graph::Graph;

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
        assert_eq!(g.get(o), -0.4620764);
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

        assert_eq!(g.get_grad(n), 0.50006473);
        assert_eq!(g.get_grad(la), 0.50006473);
        assert_eq!(g.get_grad(b), 0.50006473);
        assert_eq!(g.get_grad(l1), 0.50006473);
        assert_eq!(g.get_grad(l2), 0.50006473);
        assert_eq!(g.get_grad(x1), -1.5001942);
        assert_eq!(g.get_grad(w1), 1.0001295);
        assert_eq!(g.get_grad(x2), 0.50006473);
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
        assert_eq!(g.get(y), 0.99932873);

        g.set_grad(y, 1.);

        g.backward();

        println!("{:?}", g.nodes.get(12));

        assert_eq!(g.nodes.get(12).unwrap().grad, 0.0013420582);
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

        assert_eq!(g.get(y[0]), 0.9999091);
        assert_eq!(g.get(y[1]), 0.9999091);
        assert_eq!(g.get(y[2]), 0.9999091);
        assert_eq!(g.get(y[3]), 0.9999091);
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

        assert_eq!(g.get(y[0][0]), 0.9999091);
        assert_eq!(g.get(y[0][1]), 0.9999091);
        assert_eq!(g.get(y[0][2]), 0.9999091);
        assert_eq!(g.get(y[0][3]), 0.9999091);
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

        assert_eq!(g.get(y[0][0]), 0.9999091);
        assert_eq!(g.get(y[0][1]), 0.9999091);
        assert_eq!(g.get(y[0][2]), 0.9999091);
        assert_eq!(g.get(y[0][3]), 0.9999091);

        assert_eq!(g.get(y[1][0]), 0.99990904);
        assert_eq!(g.get(y[1][1]), 0.99990904);
        assert_eq!(g.get(y[1][2]), 0.99990904);

        assert_eq!(y[2].len(), 1);
        assert_eq!(g.get(y[2][0]), 0.9993284);
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

        for i in 0..xs.len() {
            assert_eq!(ypred[i][2].len(), 1);
        }

        assert_eq!(g.get(ypred[0][2][0]), 0.9993284);
        assert_eq!(g.get(ypred[1][2][0]), 0.9993284);
        assert_eq!(g.get(ypred[2][2][0]), 0.9993284);
        assert_eq!(g.get(ypred[3][2][0]), 0.99932826);
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
            println!("ypred {:?}", g.get_node(ypred[0][2]));

            let a = g.sub(ys, ypred[0][2]);
            let b = g.add_val(2.);
            loss = g.pow(a, b);
            let k = g.get(ys) - g.get(ypred[0][2]);

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
