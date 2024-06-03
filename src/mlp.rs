use crate::layer::Layer;

#[derive(Clone)]
pub struct MLP {
    pub ls: Vec<Layer>,
}