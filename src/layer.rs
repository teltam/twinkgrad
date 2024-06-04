use crate::graph::NodeRef;
use crate::neuron::Neuron;

#[derive(Clone)]
pub struct Layer {
    pub ls: Vec<Neuron>,
}

impl Layer {
    pub fn parameters(&self) -> Vec<NodeRef>{
        let mut params = vec!();
        for neuron in self.ls.iter() {
            params.extend(neuron.clone().parameters());
        }

        return params;
    }
}
