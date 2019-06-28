require 'json'

module Assignment2
    def assignment2?
        true
    end

    def mean(x)
      n = x.size.to_f
      u = Hash.new {|h,k| h[k] = 0.0}

      x.each do |row|
        f = row["features"]
        f.each_key do |i|
          u[i] += f[i]
        end
      end

      u.each_key {|i| u[i] /= n}
      return u
    end

    def std(x)
      u = mean(x)
      n = x.size.to_f
      s = Hash.new {|h,k| h[k] = 0.0}

      x.each do |row|
        f = row["features"]
        f.each_key do |i|
          s[i] += (f[i] - u[i]) ** 2.0
        end
      end

      s.each_key {|i| s[i] = Math.sqrt(s[i] / (n - 1))}

      return s
    end

    #Return a copy of x, so as to not change the original input
    def z_score(x)
      u = mean(x)
      s = std(x)

      nx = x.collect do |row|
        f = row["features"]
        nf = row.clone
        nf["features"] = Hash.new
        f.each_key do |i|
          nf["features"][i] = (f[i] - u[i]) / s[i]
        end

        nf
      end

      return nx
    end

    #Implement the error function given a weight vector, w
    def dot row, w
      f = row["features"]
      f.keys.inject(0.0) {|u, k| u += f[k] * w[k]}
    end

    def update_weights(w, dw, learning_rate)
      w1 = w.clone
      dw.each_key do |k|
        w1[k] -= learning_rate * dw[k]
      return w1
      end

      w1
    end

    def norm w
      0.5 * Math.sqrt(w.keys.inject(0.0) {|u,k| u += w[k] ** 2.0})
    end

    #Implementing cross validation
    def k_fold_test x, folds, eta, tol
      x =x.shuffle
      size=x.length.to_f
      datafold=x.each_slice((size/folds).ceil).to_a
      res_tr = []
      res_te = []
      iters = []
      ptr = 0
      until ptr>=datafold.size do
        train,test=train_test_split(datafold,ptr)
        g = gradient_descent(x, eta, tol)
        #list of gradient_descent returns:[iters, rmses, norms, w, zscore_bias_data],g[3] being weight w
    #     iters << g[0]
        res_tr<< rmse(test,g[3])
        res_te<< rmse(train,g[3])
        ptr+=1
      end
      result = []
      [res_tr,res_te].each do |res|
        result << { "eta" => eta, "Miu" => (mean(res)), "Std" => (std(res)) }
      end
    #   result << iters
      return result
    end


end
