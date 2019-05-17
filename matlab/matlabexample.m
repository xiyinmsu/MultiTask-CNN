
% Be aware that since Matlab is 1-indexed and column-major, 
% the usual 4 blob dimensions in Matlab are [width, height, channels, num], 
% and width is the fastest dimension. 
% Also be aware that images are in BGR channels. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% set mode and device
caffepath = '/home/yinxi/Documents/Research/caffe/';

model = [caffepath, 'models/bvlc_reference_caffenet/deploy.prototxt'];
weights = [caffepath, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'];

caffe.set_mode_gpu();
caffe.set_device(0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% create a network and access its layers and blobs
net = caffe.Net(model, weights, 'test');

% to fill blob "data" with all ones
net.blobs('data').set_data(ones(net.blobs('data').shape));

% to multiply all values in blob 'data' by 10
net.blobs('data').set_data(net.blobs('data').get_data() * 10);

% access to each layer:
% multiply conv1 parameters by 10:
net.params('conv1', 1).set_data(net.params('conv1', 1).get_data() * 10); % set weights
net.params('conv1', 2).set_data(net.params('conv1', 2).get_data() * 10); % set bias

% we could also do the following, which is the same as the above two lines
net.layers('conv1').params(1).set_data(net.layers('conv1').params(1).get_data() * 10);
net.layers('conv1').params(2).set_data(net.layers('conv1').params(2).get_data() * 10);


% to save a network you just modified:
net.save('my_net.caffemodel');

% to get a layer's type
layer_type = net.layers('conv1').type;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Forward and Backward
data = rand(net.blobs('data').shape);

res = net.forward({data});
prob = res{1};
% or
net.blobs('data').set_data(data);
net.forward_prefilled();  % takes existing data in input blobs during forward pass, takes no input and produce no output.
prob = net.blobs('prob').get_data;

% backward is similar, replace _data with _diff
% have to set force_backward: true to do backward
prob_diff = rand(nsolver = caffe.Solver([caffepath, '/models/bvlc_reference_caffenet/solver.prototxt']);
et.blobs('prob').shape);

res = net.backward({prob_diff});
data_diff = res{1};
% or
net.blobs('prob').set_diff(prob_diff);
net.backward_prefilled();
data_diff = net.blobs('data').get_diff();



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Reshape
net.blobs('data').reshape([227,227,3,1]); % run 1 image instead of 10
net.reshape();



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Training
solver = caffe.Solver([caffepath, '/models/bvlc_reference_caffenet/solver.prototxt']);

% to train
solver.solve();
% to train 1000 iterations
solver.step(1000)
% get iteration number 
iter = solver.iter();
% get its network
train_net = solver.net();
test_net = solver.test_nets(1);
% resume from a snapshot
solver.restore('your_snapshot.solverstate')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Input and output
mean_data = caffe.io.read_mean([caffepath, 'data/ilsvrc12/imagenet_mean.binaryproto']);

im_data = caffe.io.load_image([caffepath, 'examples/images/cat.jpg']);
im_data = imresize(im_data, [width, height]);
% or do it yourself:
im_data = imread([caffepath, 'examples/images/cat.jpg']);
im_data = im_data(:,:,[3,2,1]);  % caffe do BGR
im_data = permute(im_data, [2,1,3]); % permute width and height
im_data = single(im_data);   % convert to single precision



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Clear nets and solvers
caffe.reset_all()














