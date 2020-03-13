# In[27]:



c_param = [0.1, 1, 10 ,100]
gamma = [0.1, 0.5, 1]
tuned_parameters = [{'C': c_param, 'gamma': gamma}]


# In[28]:


model = SVR(kernel = 'rbf')
grid = model_selection.GridSearchCV(model, tuned_parameters, cv = kfold)


# In[29]:


grid.fit(X_train.reshape(-1,1), y_train)


# In[30]:


grid.best_params_


# In[31]:


grid.best_score_


# In[32]:


validate_pred = grid.predict(X_test.reshape(-1,1))
test_pred = grid.predict(test_X.reshape(-1,1))


# In[33]:


print('Validation')
print('<<mean_squared_error>>')
print(metrics.mean_squared_error(y_test, validate_pred))
print('<<r_squared>>')
print(metrics.r2_score(y_test, validate_pred))
print('--------------------')
print('Test')
print('<<mean_squared_error>>')
print(metrics.mean_squared_error(test_Y, test_pred))
print('<<r_squared>>')
print(metrics.r2_score(test_Y, test_pred))


# In[42]:


plt.figure(figsize = (15, 4))
plt.scatter(test_X, test_pred, color = 'r')
plt.scatter(test_X, test_Y, color = 'y')
plt.title('predict test')
plt.legend(labels = ['predict','actual'])
plt.show()


# In[43]:


plt.figure(figsize = (20, 6))
plt.scatter(X_test, validate_pred, color = 'r')
plt.scatter(X_test, y_test, color = 'y')
plt.title('predict validate')
plt.legend(labels = ['predict','actual'])
plt.show()
