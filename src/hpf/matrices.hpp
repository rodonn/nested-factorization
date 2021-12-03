#ifndef MATRICES_HPP
#define MATRICES_HPP

template<class T>
class Matrix1D {
private:
	int N;
	T *p;

public:
	Matrix1D() {
		N = 0;
		p = nullptr;
	}

	Matrix1D(int N__) {
		N = N__;
		p = new T[N];
	}

	~Matrix1D() {
		delete [] p;
	}

	inline T& get_object(int n) {
		return p[n];
	}

	inline void set_object(int n, T obj) {
		p[n] = obj;
	}

	inline T *get_pointer(int n) const {
		return(p+n);
	}

	inline int get_size1() {
		return N;
	}

	Matrix1D<T> & operator=(const Matrix1D<T> &rhs) {
    	// Check for self-assignment!
	    if (this!=&rhs) {
			delete [] p;			// deallocate memory that Matrix1D uses internally
			p = new T[rhs.N];	// Allocate memory to hod the contents of rhs
			N = rhs.N;				// Copy values from rhs
			for(int n=0; n<N; n++) {
				p[n] = *rhs.get_pointer(n);
			}
		}
		return *this;
	}
};

template<class T>
class Matrix2D {
private:
	int N;
	int M;
	T *p;

public:
	Matrix2D() {
		N = 0;
		M = 0;
		p = nullptr;
	}

	Matrix2D(int N__, int M__) {
		N = N__;
		M = M__;
		p = new T[N*M];
	}

	~Matrix2D() {
		delete [] p;
	}

	inline T& get_object(int n, int m) {
		return p[n*M+m];
	}

	inline void set_object(int n, int m, T obj) {
		p[n*M+m] = obj;
	}

	inline T *get_pointer(int n) const {
		return(p+n);
	}

	inline int get_size1() {
		return N;
	}

	inline int get_size2() {
		return M;
	}

	Matrix2D<T> & operator=(const Matrix2D<T> &rhs) {
    	// Check for self-assignment!
	    if (this!=&rhs) {
			delete [] p;			// deallocate memory that Matrix1D uses internally
			p = new T[rhs.N*rhs.M];	// Allocate memory to hod the contents of rhs
			N = rhs.N;				// Copy values from rhs
			M = rhs.M;
			for(int n=0; n<N*M; n++) {
				p[n] = *rhs.get_pointer(n);
			}
		}
		return *this;
	}
};

template<class T>
class Matrix2DNoncontiguous {
private:
	int N;
	int M;
	T **p;

public:
	Matrix2DNoncontiguous() {
		N = 0;
		M = 0;
		p = nullptr;
	}

	Matrix2DNoncontiguous(int N__, int M__) {
		N = N__;
		M = M__;
		p = new T*[N];
		for(int n=0; n<N; n++) {
			p[n] = new T[M];
		}
	}

	~Matrix2DNoncontiguous() {
		deallocate_memory();
	}

	void deallocate_memory() {
		for(int n=0; n<N; n++) {
			delete [] p[n];
		}
		delete [] p;
	}

	inline T& get_object(int n, int m) {
		return p[n][m];
	}

	inline void set_object(int n, int m, T obj) {
		p[n][m] = obj;
	}

	inline T *get_pointer(int n) const {
		return(p[n]);
	}

	inline int get_size1() {
		return N;
	}

	inline int get_size2() {
		return M;
	}

	Matrix2DNoncontiguous<T> & operator=(const Matrix2DNoncontiguous<T> &rhs) {
    	// Check for self-assignment!
	    if (this!=&rhs) {
	    	// Deallocate memory that Matrix2DNoncontiguous uses internally
			deallocate_memory();
			// Allocate memory to hod the contents of rhs
			p = new T*[rhs.N];
			for(int n=0; n<rhs.N; n++) {
				p[n] = new T[rhs.M];
			}
			// Copy values from rhs
			N = rhs.N;
			M = rhs.M;
			for(int n=0; n<N; n++) {
				T *p_aux = rhs.get_pointer(n);
				for(int m=0; m<M; m++) {
					p[n][m] = *(p_aux+m);
				}
			}
		}
		return *this;
	}
};

template<class T>
class Matrix3D {
private:
	int N;
	int M;
	int P;
	T *p;

public:
	Matrix3D() {
		N = 0;
		M = 0;
		P = 0;
		p = nullptr;
	}

	Matrix3D(int N__, int M__, int P__) {
		N = N__;
		M = M__;
		P = P__;
		p = new T[N*M*P];
	}

	~Matrix3D() {
		delete [] p;
	}

	inline T& get_object(int n, int m, int p_) {
		return p[P*M*n+P*m+p_];
	}

	inline void set_object(int n, int m, int p_, T obj) {
		p[P*M*n+P*m+p_] = obj;
	}

	inline T *get_pointer(int n) const {
		return(p+n);
	}

	inline int get_size1() {
		return N;
	}

	inline int get_size2() {
		return M;
	}

	inline int get_size3() {
		return P;
	}

	Matrix3D<T> & operator=(const Matrix3D<T> &rhs) {
    	// Check for self-assignment!
	    if (this!=&rhs) {
			delete [] p;					// deallocate memory that Matrix1D uses internally
			p = new T[rhs.N*rhs.M*rhs.P];	// Allocate memory to hod the contents of rhs
			N = rhs.N;						// Copy values from rhs
			M = rhs.M;
			P = rhs.P;
			for(int n=0; n<N*M*P; n++) {
				p[n] = *rhs.get_pointer(n);
			}
		}
		return *this;
	}
};

template<class T>
class Matrix4D {
private:
	int N;
	int M;
	int P;
	int K;
	T *p;

public:
	Matrix4D() {
		N = 0;
		M = 0;
		P = 0;
		K = 0;
		p = nullptr;
	}

	Matrix4D(int N__, int M__, int P__, int K__) {
		N = N__;
		M = M__;
		P = P__;
		K = K__;
		p = new T[N*M*P*K];
	}

	~Matrix4D() {
		delete [] p;
	}

	inline T& get_object(int n, int m, int p_, int k) {
		return p[P*M*K*n+P*K*m+K*p_+k];
	}

	inline void set_object(int n, int m, int p_, int k, T obj) {
		p[P*M*K*n+P*K*m+K*p_+k] = obj;
	}

	inline T *get_pointer(int n) const {
		return(p+n);
	}

	inline int get_size1() {
		return N;
	}

	inline int get_size2() {
		return M;
	}

	inline int get_size3() {
		return P;
	}

	inline int get_size4() {
		return K;
	}

	Matrix4D<T> & operator=(const Matrix4D<T> &rhs) {
    	// Check for self-assignment!
	    if (this!=&rhs) {
			delete [] p;					// deallocate memory that Matrix1D uses internally
			p = new T[rhs.N*rhs.M*rhs.P*rhs.K];	// Allocate memory to hod the contents of rhs
			N = rhs.N;						// Copy values from rhs
			M = rhs.M;
			P = rhs.P;
			K = rhs.K;
			for(int n=0; n<N*M*P*K; n++) {
				p[n] = *rhs.get_pointer(n);
			}
		}
		return *this;
	}
};


#endif
