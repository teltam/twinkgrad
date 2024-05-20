use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Debug, Default)]
pub struct Value<T> {
    pub data: T,
    pub _grad: f32,

    pub _prev: Vec<Value<T>>,
}

#[allow(dead_code)]
impl Value<f32> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn new_v(val: f32) -> Self {
        let mut v: Value<f32> = Default::default();
        v.data = val;
        v
    }

    pub fn _new_op(val: f32, _children: Vec<Value<f32>>) -> Self {
        let mut v: Value<f32> = Default::default();
        v.data = val;
        v._prev = _children;
        v
    }


    pub fn backward(self) -> () {
        todo!();
    }
}


// Operations
impl PartialEq for Value<f32> {
    fn eq(&self, other: &Self) -> bool {

        if self.data != other.data {
            return false;
        }

        for prev in &self._prev {
            if !other._prev.contains(prev) {
                return false;
            }
        }

        return true;
    }

    fn ne(&self, other: &Self) -> bool {
        return !self.eq(other);
    }
}

impl Value<f32> {
    pub fn to_pow(self, rhs: Value<f32>) -> Value<f32> {
        return Value::new_v(f32::powf(self.data, rhs.data));
    }
}

impl Add for Value<f32> {
    type Output = Value<f32>;

    fn add(self, rhs: Self) -> Self::Output {
        // return Value::new_v(self.data + rhs.data);
        let mut nodes = Vec::new();
        nodes.push(self.clone());
        return Value::_new_op(
            self.data + rhs.data,
            nodes
        );
    }
}

impl Sub for Value<f32> {
    type Output = Value<f32>;

    fn sub(self, rhs: Self) -> Self::Output {
        return Value::new_v(self.data - rhs.data);
    }}

impl Mul for Value<f32> {
    type Output = Value<f32>;

    fn mul(self, rhs: Self) -> Self::Output {
        return Value::new_v(self.data * rhs.data);
    }
}

impl Div for Value<f32> {
    type Output = Value<f32>;

    fn div(self, rhs: Self) -> Self::Output {
        return Value::new_v(self.data / rhs.data);
    }
}