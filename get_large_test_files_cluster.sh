# This must be ran at once whenever a clone of PHiLiP is made onto the cluster
# DESCRIPTION: Copies the large mesh files for NACA0012 that cannot be stored on GitHub

# Make user select which cluster is being used
read -p "Enter current cluster name (narval or rorqual): " CLUSTER
if [ "${CLUSTER,,}" = "narval" ]; then
    echo "Copying test files on NARVAL..."
    PATH_TO_FILES=~/projects/def-nadaraja/Libraries/TestFiles
elif [ "${CLUSTER,,}" = "rorqual" ]; then
    echo "Copying test files on RORQUAL..."
    PATH_TO_FILES=~/links/projects/def-nadaraja/Libraries/TestFiles
else 
    echo "Invalid cluster selection...Aborting."
    exit 1
fi

# Copy meshes required for integration tests
TARGET_DIR=tests/meshes/
cp ${PATH_TO_FILES}/meshes/* ${TARGET_DIR}

# Copy meshes required by unit tests
TARGET_DIR=tests/unit_tests/grid/gmsh_reader/
cp ${PATH_TO_FILES}/gmsh_reader/* ${TARGET_DIR}

# Copy initialization files for decaying homogeneous isotropic turbulence
TARGET_DIR=tests/integration_tests_control_files/decaying_homogeneous_isotropic_turbulence/setup_files/
cp ${PATH_TO_FILES}/setup_files/1proc/* ${TARGET_DIR}/1proc/
cp ${PATH_TO_FILES}/setup_files/4proc/* ${TARGET_DIR}/4proc/



