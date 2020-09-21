function theta = randInitializeTheta(m, n)
epsilon_init = 0.1;
theta = rand(m,n)*2*epsilon_init - epsilon_init;
end

