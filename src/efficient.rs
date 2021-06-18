use crate::{TreeHasher, Recording, Tree};

#[derive(Clone)]
pub struct EfficientTree<H: TreeHasher> {
    something: H::Digest
}

impl<H: TreeHasher> EfficientTree<H> {
    pub fn new(depth: usize) -> Self {
        unimplemented!()
    }
}

impl<H: TreeHasher> Tree<H> for EfficientTree<H> {
    type Recording = EfficientRecording<H>;

    fn append(&mut self, value: &H::Digest) -> bool {
        unimplemented!()
    }

    /// Obtains the current root of this Merkle tree.
    fn root(&self) -> H::Digest {
        unimplemented!()
    }

    /// Marks the current tree state leaf as a value that we're interested in
    /// witnessing. Returns true if successful and false if the tree is empty.
    fn witness(&mut self) -> bool {
        unimplemented!()
    }

    /// Obtains an authentication path to the value specified in the tree.
    /// Returns `None` if there is no available authentication path to the
    /// specified value.
    fn authentication_path(&self, value: &H::Digest) -> Option<(usize, Vec<H::Digest>)> {
        unimplemented!()
    }

    /// Marks the specified tree state value as a value we're no longer
    /// interested in maintaining a witness for. Returns true if successful and
    /// false if the value is not a known witness.
    fn remove_witness(&mut self, value: &H::Digest) -> bool {
        unimplemented!()
    }

    /// Marks the current tree state as a checkpoint if it is not already a
    /// checkpoint.
    fn checkpoint(&mut self) {
        unimplemented!()
    }

    /// Rewinds the tree state to the previous checkpoint. This function will
    /// fail and return false if there is no previous checkpoint or in the event
    /// witness data would be destroyed in the process.
    fn rewind(&mut self) -> bool {
        unimplemented!()
    }

    /// Removes the oldest checkpoint. Returns true if successful and false if
    /// there are no checkpoints.
    fn pop_checkpoint(&mut self) -> bool {
        unimplemented!()
    }

    /// Start a recording of append operations performed on a tree.
    fn recording(&self) -> EfficientRecording<H> {
        unimplemented!()
    }

    /// Plays a recording of append operations back. Returns true if successful
    /// and false if the recording is incompatible with the current tree state.
    fn play(&mut self, recording: &EfficientRecording<H>) -> bool {
        unimplemented!()
    }
}

#[derive(Clone)]
pub struct EfficientRecording<H: TreeHasher> {
    something: H::Digest
}

impl<H: TreeHasher> Recording<H> for EfficientRecording<H> {
    fn append(&mut self, value: &H::Digest) -> bool {
        unimplemented!()
    }

    fn play(&mut self, recording: &Self) -> bool {
        unimplemented!()
    }
}
