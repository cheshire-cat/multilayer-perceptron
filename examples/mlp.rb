require_relative '../lib/multilayer_perceptron'

puts 'XOR (1,0 and 0,1) MLP:'

mlp = MultilayerPerceptron.new(2, [4,1])

puts '---not trained'
puts "1,1 - #{mlp.feed([1,1]).first.round}"
puts "0,0 - #{mlp.feed([0,0]).first.round}"
puts "1,0 - #{mlp.feed([1,0]).first.round}"
puts "0,1 - #{mlp.feed([0,1]).first.round}"

10000.times do
  mlp.train([1,1],[0])
  mlp.train([0,0],[0])
  mlp.train([1,0],[1])
  mlp.train([0,1],[1])
end

puts '---trained'
puts "1,1 - #{mlp.feed([1,1]).first.round}"
puts "0,0 - #{mlp.feed([0,0]).first.round}"
puts "1,0 - #{mlp.feed([1,0]).first.round}"
puts "0,1 - #{mlp.feed([0,1]).first.round}"
