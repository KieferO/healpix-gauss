#include <Eigen/Sparse>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>  // I/O 
#include <fstream>   // file I/O
#include <iomanip>   // format manipulation
#include <string.h>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n);
void saveAsBitmap(const Eigen::VectorXd& x, int n, const char* filename);

int read_file(const char *filename, std::vector<T>& coeffs)
{
	char *line;
	char *tok;
	int i, j;
	double val;
	std::fstream infile;
	line = (char*) malloc(140);
	infile.open(filename, std::fstream::in);
	while(infile.getline(line, 139)) {
		tok = strtok(line, ",");
		i = atoi(tok);
		tok = strtok(NULL, ",");
		j = atoi(tok);
		tok = strtok(NULL, ",");
		val = atof(tok);
		if (val > 1e-10) {
			coeffs.push_back(T(i, j, val));
		}
	}
	free(line);
	return 0;
}

int main(int argc, char** argv)
{
  int i, j;
  int n = 192 * 4 * 4;  // size of the image
  int m = n*n;  // number of unknows (=number of pixels)
  FILE *outfile;

  // Assembly:
  std::vector<T> coefficients;            // list of non-zeros coefficients
  Eigen::VectorXd b(n);                   // the right hand side-vector resulting from the constraints
  Eigen::VectorXd x(n);
  read_file("cov-4.csv", coefficients);

  SpMat A(n,n);
  A.setFromTriplets(coefficients.begin(), coefficients.end());

  // Solving:
  Eigen::SimplicialCholesky<SpMat> chol(A);  // performs a Cholesky factorization of A
  outfile = fopen("invcov-4.csv", "w");
  for (i=0; i<n; i++){
    b.setZero();
    b(i) = 1.0;
    x.setZero();
    x = chol.solve(b);         // use the factorization to solve for the given right hand side
    for (j=0; j<n; j++){
      fprintf(outfile, "%d,%d,%.20e\n", i, j, x(j));
    }
  }

  return 0;
}


