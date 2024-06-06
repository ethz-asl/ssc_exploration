#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

/* Functions from libvoxel_utils.a.
 * Supports both CUDA or CPU backends
 */
extern void ComputeTSDF(double *cam_info_CPU, double *vox_info_CPU,
                        double *depth_data_CPU, double *vox_tsdf_CPU,
                        double *depth_mapping_idxs_CPU, double *occupancy);
extern void calculateOccupancyProb(double *cam_info_CPU, double *vox_info_CPU,
                                   double *depth_data_CPU,
                                   double *log_odds_occupancy);

/**
 * Python Binding functions
 * These functions call the backend CUDA or CPU functions from libvoxel_utils.a
 */
static PyObject *compute_tsdf(PyObject *self, PyObject *args) {
  PyArrayObject *cam_info_CPU, *vox_info_CPU, *depth_data_CPU, *vox_tsdf_CPU,
      *depth_mapping_idxs_CPU, *occupancy;
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!", &PyArray_Type, &cam_info_CPU,
                        &PyArray_Type, &vox_info_CPU, &PyArray_Type,
                        &depth_data_CPU, &PyArray_Type, &vox_tsdf_CPU,
                        &PyArray_Type, &depth_mapping_idxs_CPU, &PyArray_Type,
                        &occupancy)) {
    PyErr_SetString(PyExc_ValueError, "Failed to parse arguments");
    return NULL;
  }

  printf("Arguments parsed\n---------------\n");

  printf("cam_info: %d\n", cam_info_CPU->dimensions[0]);
  printf("vox_info: %d\n", vox_info_CPU->dimensions[0]);
  printf("depth_data: %d\n", depth_data_CPU->dimensions[0]);
  printf("vox_tsdf: %d\n", vox_tsdf_CPU->dimensions[0]);
  printf("depth_mapping_idxs: %d\n", depth_mapping_idxs_CPU->dimensions[0]);
  printf("occupancy: %d\n", occupancy->dimensions[0]);

  if (cam_info_CPU->nd != 1 || vox_info_CPU->nd != 1 ||
      cam_info_CPU->descr->type_num != PyArray_DOUBLE ||
      vox_info_CPU->descr->type_num != PyArray_DOUBLE ||
      depth_data_CPU->descr->type_num != PyArray_DOUBLE ||
      vox_tsdf_CPU->descr->type_num != PyArray_DOUBLE ||
      depth_mapping_idxs_CPU->descr->type_num != PyArray_DOUBLE ||
      occupancy->descr->type_num != PyArray_DOUBLE) {
    PyErr_SetString(PyExc_ValueError,
                    "arrays must be one-dimensional and of type float");
    return NULL;
  }

  ComputeTSDF((double *)cam_info_CPU->data, (double *)vox_info_CPU->data,
              (double *)depth_data_CPU->data, (double *)vox_tsdf_CPU->data,
              (double *)depth_mapping_idxs_CPU->data,
              (double *)occupancy->data);

  return Py_None;
}

static PyObject *compute_occupancy_log_prob(PyObject *self, PyObject *args) {
  PyArrayObject *cam_info_CPU, *vox_info_CPU, *depth_data_CPU,
      *occupancy_grid_CPU;
  if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &cam_info_CPU,
                        &PyArray_Type, &vox_info_CPU, &PyArray_Type,
                        &depth_data_CPU, &PyArray_Type, &occupancy_grid_CPU)) {
    PyErr_SetString(PyExc_ValueError, "Failed to parse arguments");
    return NULL;
  }

  printf("Arguments parsed\n---------------\n");

  printf("cam_info: %d\n", cam_info_CPU->dimensions[0]);
  printf("vox_info: %d\n", vox_info_CPU->dimensions[0]);
  printf("depth_data: %d\n", depth_data_CPU->dimensions[0]);
  printf("occupancy_grid: %d\n", occupancy_grid_CPU->dimensions[0]);

  if (cam_info_CPU->nd != 1 || vox_info_CPU->nd != 1 ||
      cam_info_CPU->descr->type_num != PyArray_DOUBLE ||
      vox_info_CPU->descr->type_num != PyArray_DOUBLE ||
      depth_data_CPU->descr->type_num != PyArray_DOUBLE ||
      occupancy_grid_CPU->descr->type_num != PyArray_DOUBLE) {
    PyErr_SetString(PyExc_ValueError,
                    "arrays must be one-dimensional and of type float");
    return NULL;
  }

  calculateOccupancyProb(
      (double *)cam_info_CPU->data, (double *)vox_info_CPU->data,
      (double *)depth_data_CPU->data, (double *)occupancy_grid_CPU->data);

  return Py_None;
}

static PyObject *calculate_sum(PyObject *self, PyObject *args) {
  int a, b;
  PyArg_ParseTuple(args, "ii", &a, &b);

  PyObject *value = PyLong_FromLong(a + b);
  return Py_BuildValue("N", value);
}

static PyMethodDef methods[] = { 
    {"sum", calculate_sum, METH_VARARGS, "Calculate sum"}, // (function name, function, arguments,
                                        // doc_string)
    {"compute_tsdf", compute_tsdf, METH_VARARGS,
     "Calculate TSDF from Depth Map"},
    {"compute_occupancy_log_prob", compute_occupancy_log_prob, METH_VARARGS,
     "Compute Occupancy lop probs from Depth Map"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef voxel_util_module = {
    PyModuleDef_HEAD_INIT, "VoxelUtils",  // name of the module
    "VoxelUtils", -1, methods};

PyMODINIT_FUNC PyInit_VoxelUtils(void) {
  import_array() return PyModule_Create(&voxel_util_module);
}
