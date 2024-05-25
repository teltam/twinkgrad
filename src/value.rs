use std::cell::{RefCell, RefMut};
use std::rc::Rc;

#[derive(Clone)]
pub struct Value<T: Copy>
{
    pub data: T,

    pub _grad: T,

    pub _prev: Vec<Rc<RefCell<Value<T>>>>,
    pub _op: String,

    pub _process: fn(&mut Value<f32>),
    pub _process2: fn(Rc<RefCell<Value<T>>>),
    pub _backward: fn(),
}

fn empty(_:&mut Value<f32>)  {}
fn empty2() {}
fn empty3(_: Rc<RefCell<Value<f32>>>)  {}

#[allow(dead_code)]
impl Value<f32> {
    pub fn new() -> Self {
        return Value {
            data: 0., _grad: 0., _prev: vec![], _op: "".to_string(), _process: empty,
            _process2: empty3,
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

pub fn _backward2(slf: Rc<RefCell<Value<f32>>>) {
    // let topo = &mut Vec::new();
    // let visited = &mut Vec::new();

    // without the clone and borrowing here seems to be an issue down the line.

    // (slf.clone().borrow_mut()._process2)(slf.clone());
    if slf.borrow()._op == "tanh" {
        tanh_backward2(slf.clone());
    } else if slf.borrow()._op == "**" {
        pow_backward2(slf.clone())
    } else if slf.borrow()._op == "/" {
        div_backward2(slf.clone());
    } else if slf.borrow()._op == "*" {
        mul_backward2(slf.clone());
    } else if slf.borrow()._op == "+" {
        add_backward2(slf.clone());
    } else if slf.borrow()._op == "-" {
        sub_backward2(slf.clone());
    } else {
        panic!("unsupported op");
    }

    // topo_sort(slf, topo, visited);
}

pub fn topo_sort(
    slf: Rc<RefCell<Value<f32>>>,
    topo: &mut Vec<Rc<RefCell<Value<f32>>>>,
    visited: &mut Vec<Rc<RefCell<Value<f32>>>>) {

    // for node in visited.iter() {
    //     if node.borrow().data == slf.borrow().data {
    //         return;
    //     }
    // }

    // visited.push(slf.clone());

    // for child in &slf.borrow()._prev {
    //     topo_sort(child.clone(), topo, visited);
    // }

    // After visiting all predecessors, add this node to the topological sort
    // topo.push(slf.clone());
}

// Operations
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
    out._process2 = add_backward2;

    return Rc::new(RefCell::new(out));
}

fn add_backward(slf: &mut Value<f32>) {
    println!("{}", slf._grad);
    slf._prev[0].borrow_mut()._grad = 1. * slf._grad;
    slf._prev[1].borrow_mut()._grad = 1. * slf._grad;
}

fn add_backward2(slf: Rc<RefCell<Value<f32>>>) {
    slf.borrow_mut()._prev[0].borrow_mut()._grad = 1. * slf.borrow()._grad;
    slf.borrow_mut()._prev[1].borrow_mut()._grad = 1. * slf.borrow()._grad;
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
    out._process2 = sub_backward2;

    return out;
}

fn sub_backward(slf: &mut Value<f32>) {
    println!("{}", slf._grad);
    slf._prev[0].borrow_mut()._grad = 1. * slf._grad;
    slf._prev[1].borrow_mut()._grad = -1. * slf._grad;
}

fn sub_backward2(slf: Rc<RefCell<Value<f32>>>) {
    println!("{}", slf.borrow()._grad);
    slf.borrow_mut()._prev[0].borrow_mut()._grad = 1. * slf.borrow()._grad;
    slf.borrow_mut()._prev[1].borrow_mut()._grad = -1. * slf.borrow()._grad;
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
    out._process2 = mul_backward2;

    return Rc::new(RefCell::new(out));
}

fn mul_backward(slf: &mut Value<f32>) {
    println!("{}", slf._grad);
    slf._prev[0].borrow_mut()._grad = slf._prev[1].borrow_mut().data * slf._grad;
    slf._prev[1].borrow_mut()._grad = slf._prev[0].borrow_mut().data * slf._grad;
}

fn mul_backward2(slf: Rc<RefCell<Value<f32>>>) {
    println!("{}", slf.borrow()._grad);
    slf.borrow_mut()._prev[0].borrow_mut()._grad =
        slf.borrow_mut()._prev[1].borrow_mut().data * slf.borrow()._grad;
    slf.borrow_mut()._prev[1].borrow_mut()._grad =
        slf.borrow_mut()._prev[0].borrow_mut().data * slf.borrow()._grad;
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
    out._process2 = div_backward2;

    return Rc::new(RefCell::new(out));
}

fn div_backward(slf: &mut Value<f32>) {
    println!("{}", slf._grad);
    slf._prev[0].borrow_mut()._grad = slf._prev[1].borrow_mut().data * slf._grad;
    slf._prev[1].borrow_mut()._grad = slf._prev[0].borrow_mut().data * slf._grad;
}

fn div_backward2(slf: Rc<RefCell<Value<f32>>>) {
    println!("{}", slf.borrow()._grad);
    slf.borrow_mut()._prev[0].borrow_mut()._grad =
        slf.borrow_mut()._prev[1].borrow_mut().data * slf.borrow_mut()._grad;
    slf.borrow_mut()._prev[1].borrow_mut()._grad =
        slf.borrow_mut()._prev[0].borrow_mut().data * slf.borrow_mut()._grad;
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
    out._process2 = tanh_backward2;

    return Rc::new(RefCell::new(out));
}

fn tanh_backward(slf: &mut Value<f32>) {
    slf._prev[0].borrow_mut()._grad = (1. - f32::powf(slf.data, 2.)) * slf._grad
}

fn tanh_backward2(slf: Rc<RefCell<Value<f32>>>) {
    let g = (1. - f32::powf(slf.borrow().data, 2.)) * slf.borrow()._grad;

    slf.borrow_mut()._prev[0].borrow_mut()._grad = g;
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
    out._process2 = pow_backward2;

    return Rc::new(RefCell::new(out));
}

fn pow_backward(slf: &mut Value<f32>) {
    slf._prev[0].borrow_mut()._grad = (1. - f32::powf(slf.data, 2.)) * slf._grad
}

fn pow_backward2(slf: Rc<RefCell<Value<f32>>>) {
    slf.borrow_mut()._prev[0].borrow_mut()._grad =
        (1. - f32::powf(slf.borrow().data, 2.)) * slf.borrow()._grad
}
