use crate::graph::NodeRef;
use crate::layer::Layer;

#[derive(Clone)]
pub struct MLP {
    pub ls: Vec<Layer>,
}

impl MLP {
    pub fn parameters(& self) -> Vec<NodeRef>{
        let mut params = vec!();
        for layer in self.ls.iter() {
            params.extend(layer.parameters());
        }

        return params;
    }
}
