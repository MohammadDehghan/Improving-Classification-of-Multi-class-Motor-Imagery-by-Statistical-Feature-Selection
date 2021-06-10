function data=LoadData()

    data=load('madelon_dataset.mat');

    data.nx=size(data.x,1);
    data.nt=size(data.t,1);
    data.nSample=size(data.x,2);

end