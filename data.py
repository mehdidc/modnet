class SubSample:

    def __init__(self, dataset, start ,end):
        self.dataset = dataset
        self.start = start
        self.end = end
    
    def __getitem__(self, idx):
        return self.dataset[idx + self.start]

    def __len__(self):
        return self.end - self.start 
