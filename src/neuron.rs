use crate::graph::NodeRef;

#[derive(Clone)]
pub struct Neuron {
    pub ws: Vec<NodeRef>,
    pub b: NodeRef,
}