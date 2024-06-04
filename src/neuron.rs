use crate::graph::NodeRef;

#[derive(Clone)]
pub struct Neuron {
    pub ws: Vec<NodeRef>,
    pub b: NodeRef,
}

impl Neuron {
    pub fn parameters(self) -> Vec<NodeRef>{
        let mut params = self.ws.clone();
        params.push(self.b.clone());

        return params;
    }
}