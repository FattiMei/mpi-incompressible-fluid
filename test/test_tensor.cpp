#include <Tensor.h>
#include <gtest/gtest.h>

#include <fstream>
#include <iostream>

class TensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a sample input file for loading tensor data
    std::ofstream out("in.txt");
    out << "1 2 3 4 5 6 7 8";
    out.close();

    std::ofstream out_zero("zero.txt");
    out_zero << "0 0 0 0 0 0 0 0";
    out_zero.close();
  }

  void TearDown() override {
    // Clean up the sample input file
    std::remove("in.txt");
    std::remove("zero.txt");
    std::remove("out.txt");
  }
};

TEST_F(TensorTest, LoadTensorFromFile) {
  mif::Tensor<double, 3> t({2, 2, 4});
  ASSERT_TRUE(t.load("in.txt"));
  ASSERT_EQ(t(0, 0, 0), 1);
  ASSERT_EQ(t(0, 0, 1), 2);
  ASSERT_EQ(t(0, 0, 2), 3);
  ASSERT_EQ(t(0, 0, 3), 4);
  ASSERT_EQ(t(0, 1, 0), 5);
  ASSERT_EQ(t(0, 1, 1), 6);
  ASSERT_EQ(t(0, 1, 2), 7);
  ASSERT_EQ(t(0, 1, 3), 8);
}

TEST_F(TensorTest, DumpTensorToFile) {
  mif::Tensor<double, 3> t({2, 2, 4});
  t.load("in.txt");
  ASSERT_TRUE(t.dump("out.txt"));

  std::ifstream in("out.txt");
  std::string content((std::istreambuf_iterator<char>(in)),
                      std::istreambuf_iterator<char>());
  ASSERT_EQ(content, "1 2 3 4 5 6 7 8 ");
}

TEST_F(TensorTest, AccessTensorElements) {
  mif::Tensor<double, 3> t({2, 2, 4});
  t.load("in.txt");
  ASSERT_EQ(t(0, 0, 0), 1);
  ASSERT_EQ(t(0, 1, 3), 8);
}

TEST_F(TensorTest, ModifyTensorElements) {
  mif::Tensor<double, 3> t({2, 2, 4});
  t.load("in.txt");
  t(std::integer_sequence<unsigned, 0, 0, 0>{}) = 3.14;
  ASSERT_EQ(t(std::integer_sequence<unsigned, 0, 0, 0>{}), 3.14);

  std::size_t i = 0, j = 0, k = 0;
  t(i, j, k) = 9.8;
  ASSERT_EQ(t(i, j, k), 9.8);
}

TEST_F(TensorTest, ApplyBoundaryConditions) {
  mif::Tensor<double, 3> tt({2, 2, 4});
  tt.load("zero.txt");

  auto fn = [](const unsigned i) -> double {
    if (i < 2) {
      return 2.0;
    }
    return 3.0;
  };

  for (unsigned i = 0; i < 6; ++i) {
    if (i > 0) {
      tt.apply_dirichlet_boundary_face(i - 1, 0.0);
    }
    tt.apply_dirichlet_boundary_face(i, fn, i * 0.5);
  }

  for (unsigned i = 0; i < 6; ++i) {
    tt.apply_dirichlet_boundary_face(i, 0.0);
  }
  tt.apply_dirichlet_boundary_point<int, int, int>(0, 0, 0, fn, 0);
  ASSERT_EQ(tt(0, 0, 0), fn(0));
  tt.apply_dirichlet_boundary_point<int, int, int>(1, 1, 1, fn, 10);
  ASSERT_EQ(tt(1, 1, 1), fn(10));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}