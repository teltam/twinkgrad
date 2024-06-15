use std::borrow::Cow;
use std::iter::zip;
use rand::{Rng, thread_rng};
use crate::graph::NodeType::{ComputeNode, DataNode};
use crate::layer::Layer;
use crate::mlp::MLP;
use crate::neuron::Neuron;

#[derive(Clone)]
pub struct Graph<T> {
    pub nodes: Vec<Node<T>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NodeType {
    DataNode,
    ComputeNode,
}

#[derive(Clone, Debug)]
pub struct Node<T> {
    pub label: Cow<'static, str>,
    pub data: T,

    pub grad: T,

    pub prev: Vec<NodeRef>,

    pub op: Ops,

    pub node_type: NodeType,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeRef {
    pub node_id: usize,
    pub node_type: NodeType,
}

#[derive(Clone, Debug, Copy)]
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

            node_type: DataNode,
        }
    }

    pub fn new_v(val: f64, label: String, node_type: NodeType) -> Self {
        let mut n: Node<f64> = Node::new();
        // if (val.is_nan()) {
        //     panic!();
        // }
        n.data = val;
        n.op = Ops::LEAF;
        n.label = Cow::from(label);
        n.node_type = node_type;
        n
    }

    pub fn new_op(val: f64, _prev: Vec<usize>, prev: Vec<NodeRef>, op: Ops, node_type: NodeType) -> Self {
        let mut v: Node<f64> = Node::new();
        // if (val.is_nan()) {
        //     panic!();
        // }
        v.data = val;
        v.prev = prev;
        v.op = op;
        v.node_type = node_type;
        v
    }

    pub fn new_op_l(val: f64, _prev: Vec<usize>, prev: Vec<NodeRef>, op: Ops, label: String, node_type: NodeType) -> Self {
        // if (val.is_nan()) {
        //     panic!();
        // }
        let mut v: Node<f64> = Node::new();
        v.data = val;
        v.prev = prev;
        v.op = op;
        v.label = Cow::from(label);
        v.node_type = node_type;
        v
    }
}

impl Graph<f64> {
    pub fn new() -> Self {
        Graph {
            nodes: vec![],
        }
    }
}

impl Graph<f64> {
    pub fn add_val_l(&mut self, val: f64, label: String, node_type: NodeType) -> NodeRef {
        self.nodes.push(Node::new_v(val, label, node_type));
        let id = self.nodes.len() - 1;

        NodeRef {
            node_id: id,
            node_type,
        }
    }

    pub fn add_val(&mut self, val: f64, node_type: NodeType) -> NodeRef {
        self.nodes.push(Node::new_v(val, "".to_string(), node_type));
        let id = self.nodes.len() - 1;

        NodeRef {
            node_id: id,
            node_type,
        }
    }

    pub fn a_range(&mut self, l: u8) -> Vec<NodeRef> {
        let mut res = vec!();
        for i in 0..l {
            res.push(self.add_val_l(f64::from(i), "".to_string(), DataNode));
        }

        res
    }

    pub fn add_val_prev(
        &mut self,
        val: f64,
        _prev: Vec<usize>,
        prev: Vec<NodeRef>,
        op: Ops,
        node_type: NodeType,
    ) -> NodeRef {

        self.nodes.push(Node::new_op(val, _prev, prev, op, node_type));
        let id = self.nodes.len() - 1;

        NodeRef {
            node_id: id,
            node_type,
        }
    }

    pub fn add_val_prev_l(
        &mut self,
        val: f64,
        _prev: Vec<usize>,
        prev: Vec<NodeRef>,
        op: Ops,
        label: String,
        node_type: NodeType
    ) -> NodeRef {
        self.nodes.push(Node::new_op_l(val, _prev, prev, op, label, node_type));
        let id = self.nodes.len() - 1;

        NodeRef {
            node_id: id,
            node_type,
        }
    }

    pub fn neuron(&mut self, l: usize) -> Neuron {
        let mut rng = thread_rng();

        let mut ws = vec!();
        for i in 0..l {
            ws.push(self.add_val_l(rng.gen::<f64>(), format!("w_{}", i), DataNode));
        }

        let b = self.add_val_l(rng.gen::<f64>(), "b".to_string(), DataNode);

        Neuron { ws, b, }
    }

    pub fn neuron_ones(&mut self, l: usize) -> Neuron {
        let mut ws = vec!();
        for i in 0..l {
            ws.push(self.add_val_l(1., format!("w_{}", i), DataNode));
        }

        let b = self.add_val_l(1., "b".to_string(), DataNode);

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
        return self.get_node(x).grad;
    }

    pub fn set_grad(&mut self, x: NodeRef, grad: f64) {
        self.nodes.get_mut(x.node_id).unwrap().grad = grad;
    }

    pub fn set_data(&mut self, x: NodeRef, data: f64) {
        self.nodes.get_mut(x.node_id).unwrap().data = data;
    }

    // Compute Operations
    pub fn add(&mut self, x1: NodeRef, x2: NodeRef, label: String) -> NodeRef {
        // let mut a1 = self.get_node_mut(x1);
        // let mut a2 = self.get_node_mut(x2);
        let a1 = self.nodes.get_mut(x1.node_id).unwrap().data;
        let a2 = self.nodes.get_mut(x2.node_id).unwrap().data;

        // let a1 = self.get_node(x1).data;
        // let a2 = self.get_node(x2).data;

        self.add_val_prev_l(
            a1 + a2,
            vec![x1.node_id, x2.node_id],
            vec![x1, x2],
            Ops::ADD,
            label,
            ComputeNode,
        );

        NodeRef {
            node_id: self.nodes.len() - 1,
            node_type: ComputeNode,
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
            ComputeNode,
        );

        NodeRef {
            node_id: self.nodes.len() - 1,
            node_type: ComputeNode,
        }
    }

    pub fn mul(&mut self, x1: NodeRef, x2: NodeRef) -> NodeRef {
        // let a1 = self.nodes.get_mut(x1.node_id).unwrap().data;
        // let a2 = self.nodes.get_mut(x2.node_id).unwrap().data;
        let a1 = self.get_node(x1).data;
        let a2 = self.get_node(x2).data;

        self.add_val_prev(
            a1 * a2,
            vec![x1.node_id, x2.node_id],
            vec![x1, x2],
            Ops::MUL,
            ComputeNode,
        );

        NodeRef {
            node_id: self.nodes.len() - 1,
            node_type: ComputeNode,
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
            ComputeNode,
        );

        NodeRef {
            node_id: self.nodes.len() - 1,
            node_type: ComputeNode,
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
            ComputeNode,
        );

        NodeRef {
            node_id: self.nodes.len() - 1,
            node_type: ComputeNode,
        }
    }

    pub fn tanh(&mut self, x1: NodeRef) -> NodeRef {
        // let a1 = self.nodes.get_mut(x1.node_id).unwrap().data;
        let a1 = self.get_node(x1).data;

        let t = (f64::powf(2.718, 2. * a1) - 1.) / (f64::powf(2.718, 2. * a1) + 1.);

        self.add_val_prev(t, vec![x1.node_id], vec![x1], Ops::TANH, ComputeNode);

        // if t.is_nan() {
        //     println!("---- in tanh {:?}, {:?}, {}, {}", self.get_node(x1), t, (f64::powf(2.718, 2. * a1) - 1.), (f64::powf(2.718, 2. * a1) + 1.));
        // }

        NodeRef {
            node_id: self.nodes.len() - 1,
            node_type: ComputeNode
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

    pub fn apply_mlp(&mut self, mlp: &MLP, x: &Vec<NodeRef>) -> Vec<NodeRef> {
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

        // return res[1..res.len()].to_vec();
        return res.last().unwrap().clone();
    }

    // ==== Backward methods ====

    pub fn backward(&mut self) {
        let mut node_list = self.nodes.clone();

        self.backward_retain_graph();

        // at this point mark the nodes.
        let mut c = 0;
        for node in &node_list {
            if node.node_type == ComputeNode {
                c += 1;
            }
        }
        println! {"Counter: {}", c}
        node_list.retain(|i| i.node_type == DataNode);
        c = 0;
        for node in &node_list {
            if node.node_type == ComputeNode {
                c += 1;
            }
        }
        println! {"Counter: {}", c}
        self.nodes = node_list;
    }

    pub fn backward_retain_graph(&mut self) {
        let mut node_list = self.nodes.clone();

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
