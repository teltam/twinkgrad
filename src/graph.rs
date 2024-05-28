use std::borrow::Cow;

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

impl Node<f32> {
    pub fn new() -> Self {
        Node {
            data: 0.,
            grad: 0.,
            prev: vec![],
            op: Ops::EMPTY,

            label: Cow::from(""),
        }
    }

    pub fn new_v(val: f32, label: String) -> Self {
        let mut n: Node<f32> = Node::new();
        n.data = val;
        n.op = Ops::LEAF;
        n.label = Cow::from(label);
        n
    }

    pub fn new_op(val: f32, _prev: Vec<usize>, prev: Vec<NodeRef>, op: Ops) -> Self {
        let mut v: Node<f32> = Node::new();
        v.data = val;
        v.prev = prev;
        v.op = op;
        v
    }

    pub fn new_op_l(
        val: f32,
        _prev: Vec<usize>,
        prev: Vec<NodeRef>,
        op: Ops,
        label: String,
    ) -> Self {
        let mut v: Node<f32> = Node::new();
        v.data = val;
        v.prev = prev;
        v.op = op;
        v.label = Cow::from(label);
        v
    }
}

impl Graph<f32> {
    pub fn new() -> Self {
        Graph { nodes: vec![] }
    }
}

impl Graph<f32> {
    pub fn add_val(&mut self, val: f32, label: String) -> NodeRef {
        self.nodes.push(Node::new_v(val, label));

        NodeRef {
            node_id: self.nodes.len() - 1,
        }
    }

    pub fn add_val_prev(
        &mut self,
        val: f32,
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
        val: f32,
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

    pub fn get(&mut self, x: NodeRef) -> f32 {
        let node_id = x.node_id;

        return self.nodes.get(node_id).unwrap().data;
    }

    pub fn get_grad(&mut self, x: NodeRef) -> f32 {
        let node_id = x.node_id;

        return self.nodes.get(node_id).unwrap().grad;
    }

    pub fn set_grad(&mut self, x: NodeRef, grad: f32) {
        let node_id = x.node_id;

        self.nodes.get_mut(node_id).unwrap().grad = grad;
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
            f32::powf(a1, a2),
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

        let t = (f32::powf(2.718, 2. * a1) - 1.) / (f32::powf(2.718, 2. * a1) + 1.);

        self.add_val_prev(t, vec![x1.node_id], vec![x1], Ops::TANH);

        NodeRef {
            node_id: self.nodes.len() - 1,
        }
    }

    pub fn backward(&mut self) {
        let node_list = self.nodes.clone();

        println!("{:?}", node_list);

        // already topo sorted
        for (i, node) in node_list.iter().rev().enumerate() {
            let ii = node_list.len() - 1 - i;

            println!("{:?}, {:?}", ii, node);

            match &node.op {
                Ops::TANH => self.tanh_backward(ii),
                Ops::EXP => self.pow_backward(ii),
                Ops::DIV => self.div_backward(ii),
                Ops::MUL => self.mul_backward(ii),
                Ops::ADD => self.add_backward(ii),
                Ops::SUB => self.sub_backward(ii),
                Ops::LEAF => return,
                Ops::EMPTY => {
                    panic!(
                        "unsupported op node_label: {}, op: {:?}",
                        node.label, node.op
                    )
                }
            }
        }

        self.nodes = node_list;
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
        c2_node.grad += (-c1_node.data / f32::powf(c2_node.data, 2.)) * node.grad;

        self.nodes[c1.node_id] = c1_node;
        self.nodes[c2.node_id] = c2_node;
    }

    fn tanh_backward(&mut self, i: usize) {
        let node = self.nodes.get(i).unwrap().clone();

        let c1 = node.prev.get(0).unwrap();

        let mut c1_node = self.nodes.get_mut(c1.node_id).unwrap().clone();

        let g = (1. - f32::powf(node.data, 2.)) * node.grad;

        c1_node.grad += g;

        self.nodes[c1.node_id] = c1_node;
    }

    fn pow_backward(&mut self, i: usize) {
        let node = self.nodes.get(i).unwrap().clone();

        let c1 = node.prev.get(0).unwrap();
        let c2 = node.prev.get(1).unwrap();

        let mut c1_node = self.nodes.get_mut(c1.node_id).unwrap().clone();
        let c2_node = self.nodes.get_mut(c2.node_id).unwrap().clone();

        let g = c2_node.data * f32::powf(c1_node.data, c2_node.data - 1.) * node.grad;

        c1_node.grad += g;

        self.nodes[c1.node_id] = c1_node;
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::Graph;

    #[test]
    fn graph_op() {
        let g = &mut Graph::new();

        let x1 = g.add_val(3., "x1".to_string());
        let x2 = g.add_val(0.5, "x2".to_string());

        let w1 = g.add_val(-3.0, "w1".to_string());
        let w2 = g.add_val(1., "w2".to_string());

        let b = g.add_val(8., "b".to_string());

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

        let x1 = g.add_val(2., "x1".to_string());
        let x2 = g.add_val(0.0, "x2".to_string());

        let w1 = g.add_val(-3., "w1".to_string());
        let w2 = g.add_val(1., "w2".to_string());

        let b = g.add_val(6.8813735870195432, "b".to_string());

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
}
