# In[16]:


c_val = 1000
gmm = 0.1
model_li = LinearRegression()
model_svr_li = SVR(kernel='linear', C=c_val)
model_svr_rbf = SVR(kernel='rbf', C=c_val, gamma=gmm)
model_svr_pol = SVR(kernel='poly', C=c_val, degree=2)
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state = seed, shuffle=True)
score_model_li = model_selection.cross_val_score(model_li, X_train.reshape(-1,1), y_train, cv = kfold)
score_svr_li = model_selection.cross_val_score(model_svr_li, X_train.reshape(-1,1), y_train, cv = kfold)
score_svr_rbf = model_selection.cross_val_score(model_svr_rbf, X_train.reshape(-1,1), y_train, cv = kfold)
score_svr_pol = model_selection.cross_val_score(model_svr_pol, X_train.reshape(-1,1), y_train, cv = kfold)


# In[17]:


print(score_model_li)
print(score_svr_li)
print(score_svr_rbf)
print(score_svr_pol)


# In[18]:


# fig = plt.figure(figsize = (20, 10))

# plt.plot(score_model_li, marker = 'o', color = 'r')
# plt.plot(score_svr_li, marker = 'o', color = 'g')
# plt.plot(score_svr_rbf, marker = 'o', color = 'b')
# plt.plot(score_svr_pol, marker = 'o', color = 'c')
# plt.title('score')
# plt.show()
fig = plt.figure(figsize = (20,10))
arr = np.arange(10)
plt.bar(arr+0, score_model_li, color = 'r', width = 0.25)
plt.bar(arr+0.25, score_svr_li, color = 'g', width = 0.25)
plt.bar(arr+0.5, score_svr_rbf, color = 'b', width = 0.25)
plt.bar(arr+0.75, score_svr_pol, color = 'c', width = 0.25)
plt.legend(labels = ['score_model_li','score_svr_li','score_svr_rbf','score_svr_pol'])
plt.show()


# In[19]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['score_model_li', 'score_svr_li', 'score_svr_rbf', 'score_svr_pol']
students = [score_model_li.mean(),score_svr_li.mean(),score_svr_rbf.mean(),score_svr_pol.mean()]
ax.bar(langs,students)
plt.show()


# In[20]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['score_model_li', 'score_svr_li', 'score_svr_rbf', 'score_svr_pol']
students = [score_model_li.std(),score_svr_li.std(),score_svr_rbf.std(),score_svr_pol.std()]
ax.bar(langs,students)
plt.show()


# In[21]:


model_li.fit(X_train.reshape(-1,1), y_train)
model_svr_li.fit(X_train.reshape(-1,1), y_train)
model_svr_pol.fit(X_train.reshape(-1,1), y_train)
model_svr_rbf.fit(X_train.reshape(-1,1), y_train)


# In[22]:


li_pred = model_li.predict(test_X.reshape(-1,1))
svr_li_pred = model_svr_li.predict(test_X.reshape(-1,1))
svr_rbf_pred = model_svr_rbf.predict(test_X.reshape(-1,1))
svr_pol_pred = model_svr_pol.predict(test_X.reshape(-1,1))

li_pred_v = model_li.predict(X_test.reshape(-1,1))
svr_li_pred_v = model_svr_li.predict(X_test.reshape(-1,1))
svr_rbf_pred_v = model_svr_rbf.predict(X_test.reshape(-1,1))
svr_pol_pred_v = model_svr_pol.predict(X_test.reshape(-1,1))


# In[23]:


print('Validation')
print('<<mean_squared_error>>')
print('linear')
print(metrics.mean_squared_error(y_test, li_pred_v))
print('svr linear')
print(metrics.mean_squared_error(y_test, svr_li_pred_v))
print('svr rbf')
print(metrics.mean_squared_error(y_test, svr_rbf_pred_v))
print('svr poly')
print(metrics.mean_squared_error(y_test, svr_pol_pred_v))
print('<<r2_score>>')
print('linear')
print(metrics.r2_score(y_test, li_pred_v))
print('svr linear')
print(metrics.r2_score(y_test, svr_li_pred_v))
print('svr rbf')
print(metrics.r2_score(y_test, svr_rbf_pred_v))
print('svr poly')
print(metrics.r2_score(y_test, svr_pol_pred_v))


# In[24]:


print('Test')
print('<<mean_squared_error>>')
print('linear')
print(metrics.mean_squared_error(test_Y, li_pred))
print('svr linear')
print(metrics.mean_squared_error(test_Y, svr_li_pred))
print('svr rbf')
print(metrics.mean_squared_error(test_Y, svr_rbf_pred))
print('svr poly')
print(metrics.mean_squared_error(test_Y, svr_pol_pred))
print('<<r2_score>>')
print('linear')
print(metrics.r2_score(test_Y, li_pred))
print('svr linear')
print(metrics.r2_score(test_Y, svr_li_pred))
print('svr rbf')
print(metrics.r2_score(test_Y, svr_rbf_pred))
print('svr poly')
print(metrics.r2_score(test_Y, svr_pol_pred))


# In[25]:


plt.figure(figsize = (20, 6))

plt.scatter(X_test, li_pred_v, color = 'g')
plt.scatter(X_test, svr_li_pred_v, color = 'b')
plt.scatter(X_test, svr_rbf_pred_v, color = 'c')
plt.scatter(X_test, svr_pol_pred_v, color = 'y')
plt.title('predict validation')
plt.legend(labels = ['model_li','svr_li','svr_rbf','svr_pol'])
plt.show()


# In[26]:


plt.figure(figsize = (20, 6))

plt.scatter(test_X, li_pred, color = 'g')
plt.scatter(test_X, svr_li_pred, color = 'b')
plt.scatter(test_X, svr_rbf_pred, color = 'c')
plt.scatter(test_X, svr_pol_pred, color = 'y')
plt.title('predict test')
plt.legend(labels = ['model_li','svr_li','svr_rbf','svr_pol'])
plt.show()