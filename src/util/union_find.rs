use std::collections::HashMap;
use std::hash::Hash;

pub struct UnionFind<V> {
    nodes: Vec<Node<V>>,
    indices: HashMap<V, Index>,
}

type Index = usize;
struct Node<V> {
    content: V,
    p: Option<Index>,
    size: usize,
}

impl<V> Node<V> {
    fn parent<'a, 'b>(&'a self, nodes: &'b [Node<V>]) -> Option<&Node<V>>
    where
        'b: 'a,
    {
        if let Some(p) = self.p {
            Some(&nodes[p])
        } else {
            None
        }
    }
    fn parent_mut<'a, 'b>(&'a self, nodes: &'b mut [Node<V>]) -> Option<&mut Node<V>>
    where
        'b: 'a,
    {
        if let Some(p) = self.p {
            Some(&mut nodes[p])
        } else {
            None
        }
    }
}
impl<V> From<V> for Node<V> {
    fn from(v: V) -> Self {
        Self {
            content: v,
            p: None,
            size: 1,
        }
    }
}

impl<V> UnionFind<V> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            indices: HashMap::new(),
        }
    }
}

impl<V> Default for UnionFind<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> UnionFind<V>
where
    V: Eq + Hash + Clone,
{
    pub fn new_set(&mut self, node: V) {
        if self.indices.contains_key(&node) {
            return;
        }
        self.nodes.push(Node::from(node.clone()));
        self.indices.insert(node, self.nodes.len() - 1);
    }
    pub fn find(&mut self, node: &V) -> Option<&V> {
        if let Some(&i) = self.indices.get(node) {
            let res = self.find_rec(i);
            Some(&self.nodes[res].content)
        } else {
            None
        }
    }
    pub fn union(&mut self, node1: &V, node2: &V) {
        if let (Some(&n1), Some(&n2)) = (self.indices.get(node1), self.indices.get(node2)) {
            let p1_i = self.find_rec(n1);
            let p2_i = self.find_rec(n2);
            if p1_i == p2_i {
                return;
            }
            let p1 = &self.nodes[p1_i];
            let p2 = &self.nodes[p2_i];
            let p2_size = p2.size;
            let p1_size = p1.size;
            if p1.size > p2_size {
                self.nodes[p2_i].p = Some(p1_i);
                self.nodes[p1_i].size += p2_size;
            } else {
                self.nodes[p1_i].p = Some(p2_i);
                self.nodes[p2_i].size += p1_size;
            }
        }
    }
    fn find_rec(&mut self, index: Index) -> Index {
        let node = &self.nodes[index];
        let parent_index = node.p;
        if let Some(p) = parent_index {
            let new_parent = Some(self.find_rec(p));
            let node = &mut self.nodes[index];
            node.p = new_parent;
            node.p.unwrap()
        } else {
            index
        }
    }
}
