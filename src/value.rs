use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;
use std::borrow::Borrow;
use std::mem::ManuallyDrop;

#[derive(Clone)]
pub struct Value<T: Copy>
{
    pub data: T,
    pub _grad: T,

    pub _prev: Vec<Value<T>>,
    pub _op: String,

    // pub _backward: Rc<dyn FnMut(Value<T>, Value<T>) -> ()>,
    pub _backward: ManuallyDrop<Rc<dyn FnMut() -> ()>>,
}

#[allow(dead_code)]
impl Value<f32> {
    pub fn new() -> Self {
        return Value {
            data: 0., _grad: 0., _prev: vec![],
            _op: "".to_string(),
            _backward: ManuallyDrop::new(Rc::new(|| {})),
        }
    }

    pub fn new_v(val: f32) -> Self {
        let mut v: Value<f32> = Value::new();
        v.data = val;
        v
    }

    pub fn new_op(val: f32, _children: Vec<Value<f32>>, _op: String) -> Self {
        let mut v: Value<f32> = Value::new();
        v.data = val;
        v._prev = _children;
        v._op = _op;
        v
    }

    pub fn clone(&self) -> Value<f32> {
        Value {
            data: self.data,
            _grad: self._grad,
            _prev: vec![],
            _op: "".to_string(),
            _backward: ManuallyDrop::new(Rc::new(|| {})), // Placeholder for backward function
        }
    }
}


// Operations
impl PartialEq for Value<f32> {
    fn eq(&self, other: &Self) -> bool {
        // TODO this is broken until I figure the best way to cmp two ASTs with self-ref.

        if self.data != other.data {
            return false;
        }

        return true;
    }

    fn ne(&self, other: &Self) -> bool {
        return !self.eq(other);
    }
}

impl Value<f32> {
    pub fn to_pow(self, rhs: Value<f32>) -> Value<f32> {
        let mut nodes = Vec::new();

        let a = self.data;
        let b = rhs.data;
        nodes.push(self);
        nodes.push(rhs);

        return Value::new_op(
            f32::powf(a, b),
            nodes,
            "**".to_string(),
        );
    }

    pub fn tanh(mut self) -> Value<f32> {
        let n = self.data;
        let t = (f32::powf(2.718, 2. * n) - 1.) / (f32::powf(2.718, 2. * n) + 1.);

        let mut nodes = vec!(self);

        let mut out = Value::new_op(
            t,
            nodes.clone(),
            "tanh".to_string(),
        );

        out._backward = ManuallyDrop::new (Rc::new(|| {
            nodes[0]._grad = (1. - f32::powf(t, 2.)) * out._grad;
        }));

        return out;
    }
}


impl Add for Value<f32> {
    type Output = Value<f32>;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        let mut nodes = Vec::new();

        let a = self.data;
        let b = rhs.data;
        nodes.push(self);
        nodes.push(rhs);

        let mut out = Value::new_op(
            a + b,
            nodes,
            "+".to_string(),
        );

        out._backward = ManuallyDrop::new(Rc::new(|| {
            self._grad = 1. * out._grad;
            rhs._grad = 1. * out._grad;
        }));

        return out;
    }
}

impl Sub for Value<f32> {
    type Output = Value<f32>;

    fn sub(mut self, mut rhs: Self) -> Self::Output {
        let mut nodes = Vec::new();

        let a = self.data;
        let b = rhs.data;

        nodes.push(self);
        nodes.push(rhs);

        let mut out = Value::new_op(
            a - b,
            nodes,
            "-".to_string(),
        );

        out._backward = ManuallyDrop::new(Rc::new(|| {
            self._grad = 1. * out._grad;
            rhs._grad = -1. * out._grad;
        }));

        return out;
    }}

impl Mul for Value<f32> {
    type Output = Value<f32>;

    fn mul(mut self, mut rhs: Self) -> Self::Output {
        let mut nodes = Vec::new();

        let a = self.data;
        let b = rhs.data;

        nodes.push(self);
        nodes.push(rhs);

        let mut out = Value::new_op(
            a * b,
            nodes,
            "*".to_string(),
        );

        out._backward = ManuallyDrop::new(Rc::new(|| {
            self._grad = b * out._grad;
            rhs._grad = a * out._grad;
        }));

        return out;
    }
}

impl Div for Value<f32> {
    type Output = Value<f32>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut nodes = Vec::new();

        let a = self.data;
        let b = rhs.data;

        nodes.push(self);
        nodes.push(rhs);

        return Value::new_op(
            a / b,
            nodes,
            "/".to_string(),
        );
    }
}