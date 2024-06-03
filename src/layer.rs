use crate::neuron::Neuron;

#[derive(Clone)]
pub struct Layer {
    pub ls: Vec<Neuron>,
}