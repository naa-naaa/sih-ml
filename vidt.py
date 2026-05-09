from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pickle
model = pickle.load(open('best_model.pkl', 'rb'))
encoder = pickle.load(open('label_encoder_best.pkl', 'rb'))

plt.figure(figsize=(20, 10))
plot_tree(model, 
          feature_names=['gwl_avg', 'trend_slope', 'rainfall_avg', 'recharge_avg'],
          class_names=encoder.classes_,
          filled=True,
          rounded=True,
          fontsize=10)
plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
plt.show()