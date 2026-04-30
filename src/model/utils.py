from ase import Atoms


def Batch2Atoms(batch):
    images = []
    for i in range(batch.num_graphs):
        atoms = Atoms(
            numbers=batch.get_example(i).atomic_numbers.numpy(),
            positions=batch.get_example(i).pos.numpy(),
        )
        images.append(atoms)

    return images
