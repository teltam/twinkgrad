use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone)]
pub struct Value<T: Copy>
{
    pub data: T,

    pub _index: u8, // index of the node itself.

    pub _grad: T,

    pub _prev: Vec<Rc<RefCell<Value<T>>>>,
    pub _op: String,

    pub _process: fn(&mut Value<f32>),
    pub _backward: fn(),
}

fn empty(_:&mut Value<f32>)  {}
fn empty2() {}

#[allow(dead_code)]
impl Value<f32> {
    pub fn new() -> Self {
        return Value {
            data: 0., _index:0, _grad: 0., _prev: vec![], _op: "".to_string(), _process: empty,
            _backward: empty2,
        }
    }

    pub fn new_v(val: f32) -> Self {
        let mut v: Value<f32> = Value::new();
        v.data = val;
        v
    }

    pub fn new_op(val: f32, _children: Vec<Rc<RefCell<Value<f32>>>>, _op: String) -> Self {
        let mut v: Value<f32> = Value::new();
        v.data = val;
        v._prev = _children;
        v._op = _op;
        v
    }

    pub fn _backward(&mut self) {
        (self._process)(self);
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

pub fn special_add(a: Rc<RefCell<Value<f32>>>, b: Rc<RefCell<Value<f32>>>)
    -> Rc<RefCell<Value<f32>>> {
    let mut nodes = Vec::new();

    let sum = a.borrow_mut().data + b.borrow_mut().data;

    nodes.push(a.clone());
    nodes.push(b.clone());

    let mut out = Value::new_op(
        sum,
        nodes,
        "+".to_string(),
    );

    out._process = add_backward;

    return Rc::new(RefCell::new(out));
}

fn add_backward(slf: &mut Value<f32>) {
    println!("{}", slf._grad);
    slf._prev[0].borrow_mut()._grad = 1. * slf._grad;
    slf._prev[1].borrow_mut()._grad = 1. * slf._grad;
}

pub fn special_sub(a: Rc<RefCell<Value<f32>>>, b: Rc<RefCell<Value<f32>>>) -> Value<f32> {
    let mut nodes = Vec::new();

    let sum = a.borrow_mut().data - b.borrow_mut().data;

    nodes.push(a.clone());
    nodes.push(b.clone());

    let mut out = Value::new_op(
        sum,
        nodes,
        "-".to_string(),
    );

    out._process = sub_backward;

    return out;
}

fn sub_backward(slf: &mut Value<f32>) {
    println!("{}", slf._grad);
    slf._prev[0].borrow_mut()._grad = 1. * slf._grad;
    slf._prev[1].borrow_mut()._grad = -1. * slf._grad;
}

pub fn special_mul(a: Rc<RefCell<Value<f32>>>, b: Rc<RefCell<Value<f32>>>)
    ->  Rc<RefCell<Value<f32>>> {
    let mut nodes = Vec::new();

    let sum = a.borrow_mut().data * b.borrow_mut().data;

    nodes.push(a.clone());
    nodes.push(b.clone());

    let mut out = Value::new_op(
        sum,
        nodes,
        "*".to_string(),
    );

    out._process = mul_backward;

    return Rc::new(RefCell::new(out));
}

fn mul_backward(slf: &mut Value<f32>) {
    println!("{}", slf._grad);
    slf._prev[0].borrow_mut()._grad = slf._prev[1].borrow_mut().data * slf._grad;
    slf._prev[1].borrow_mut()._grad = slf._prev[0].borrow_mut().data * slf._grad;
}


pub fn special_div(a: Rc<RefCell<Value<f32>>>, b: Rc<RefCell<Value<f32>>>)
                   ->  Rc<RefCell<Value<f32>>> {
    let mut nodes = Vec::new();

    let sum = a.borrow_mut().data / b.borrow_mut().data;

    nodes.push(a.clone());
    nodes.push(b.clone());

    let mut out = Value::new_op(
        sum,
        nodes,
        "/".to_string(),
    );

    out._process = div_backward;

    return Rc::new(RefCell::new(out));
}

fn div_backward(slf: &mut Value<f32>) {
    println!("{}", slf._grad);
    slf._prev[0].borrow_mut()._grad = slf._prev[1].borrow_mut().data * slf._grad;
    slf._prev[1].borrow_mut()._grad = slf._prev[0].borrow_mut().data * slf._grad;
}

pub fn special_tanh(slf: Rc<RefCell<Value<f32>>>) -> Rc<RefCell<Value<f32>>> {
    let n = slf.borrow_mut().data;
    let t = (f32::powf(2.718, 2. * n) - 1.) / (f32::powf(2.718, 2. * n) + 1.);

    let nodes = vec!(slf.clone());

    let mut out = Value::new_op(
        t,
        nodes,
        "tanh".to_string(),
    );

    out._process = tanh_backward;

    return Rc::new(RefCell::new(out));
}

fn tanh_backward(slf: &mut Value<f32>) {
    slf._prev[0].borrow_mut()._grad = (1. - f32::powf(slf.data, 2.)) * slf._grad
}

pub fn special_pow(a: Rc<RefCell<Value<f32>>>, b: Rc<RefCell<Value<f32>>>)
                   ->  Rc<RefCell<Value<f32>>> {

    let mut nodes = Vec::new();

    let pow = f32::powf(a.borrow_mut().data, b.borrow_mut().data);

    nodes.push(a.clone());
    nodes.push(b.clone());

    let mut out = Value::new_op(
        pow,
        nodes,
        "**".to_string(),
    );

    out._process = pow_backward;

    return Rc::new(RefCell::new(out));
}

fn pow_backward(slf: &mut Value<f32>) {
    // TODO fix
    slf._prev[0].borrow_mut()._grad = (1. - f32::powf(slf.data, 2.)) * slf._grad
}