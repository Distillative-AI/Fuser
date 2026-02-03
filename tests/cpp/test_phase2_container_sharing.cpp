// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "fusion.h"
#include "ir/container.h"
#include "ops/all_ops.h"
#include "tests/cpp/utils.h"

namespace nvfuser {

// Test class for Phase 2 container sharing tests
class Phase2ContainerTest : public NVFuserTest {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();
  }
  void TearDown() override {
    NVFuserTest::TearDown();
  }
};

// =============================================================================
// Task 1 Tests: Locking Infrastructure
// =============================================================================

TEST_F(Phase2ContainerTest, LockingBasic) {
  // Verify basic operations still work with locking in place
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  // Verify container has expected contents
  // Use vals() and unordered_exprs() which return references to container data
  EXPECT_GT(fusion.vals().size(), 0);
  EXPECT_GT(fusion.unordered_exprs().size(), 0);
}

TEST_F(Phase2ContainerTest, ConcurrentReads) {
  // Multiple threads can read simultaneously without data races
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  std::vector<std::thread> threads;
  std::atomic<int> read_count{0};

  // Spawn multiple reader threads
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < 100; ++j) {
        // Access vals and unordered_exprs through fusion's forwarding methods
        // These return const references to the underlying container data
        const auto& vals = fusion.vals();
        const auto& exprs = fusion.unordered_exprs();
        // Just access sizes to verify no crashes under concurrent access
        (void)vals.size();
        (void)exprs.size();
        read_count++;
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(read_count.load(), 400);
}

// =============================================================================
// Task 2 Tests: Fusion Tracking Infrastructure
// =============================================================================

TEST_F(Phase2ContainerTest, FusionRegistration) {
  // Test that addFusion increments count, removeFusion decrements
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Get the IrContainer through Fusion
  auto& container = *fusion.ir_container();

  // With Phase 2, Fusion constructor calls addFusion(this), so count starts at
  // 1
  EXPECT_EQ(container.sharingCount(), 1);
  EXPECT_FALSE(container.hasMultipleFusions());

  // Register another Fusion with the same container
  // (simulating shared_ptr sharing that will happen in later tasks)
  Fusion fusion2;
  container.addFusion(&fusion2);
  EXPECT_EQ(container.sharingCount(), 2);
  EXPECT_TRUE(container.hasMultipleFusions());

  // Remove the manually added one
  container.removeFusion(&fusion2);
  EXPECT_EQ(container.sharingCount(), 1);
  EXPECT_FALSE(container.hasMultipleFusions());

  // Note: Don't remove fusion here - its destructor will handle that
  // If we remove it manually, the destructor will try to remove again
}

TEST_F(Phase2ContainerTest, FusionTransfer) {
  // Test transferFusion correctly updates tracking
  // Note: We use an IrContainer directly to test the transfer mechanism
  // without interference from Fusion's auto-registration

  auto container = std::make_shared<IrContainer>();

  // Create dummy fusion pointers for testing (not real Fusions)
  Fusion fusion1;
  Fusion fusion2;

  // Register fusion1 with test container
  container->addFusion(&fusion1);
  EXPECT_EQ(container->sharingCount(), 1);
  EXPECT_TRUE(container->sharingFusions().count(&fusion1) > 0);
  EXPECT_TRUE(container->sharingFusions().count(&fusion2) == 0);

  // Transfer from fusion1 to fusion2
  container->transferFusion(&fusion1, &fusion2);
  EXPECT_EQ(container->sharingCount(), 1);
  EXPECT_TRUE(container->sharingFusions().count(&fusion1) == 0);
  EXPECT_TRUE(container->sharingFusions().count(&fusion2) > 0);

  // Clean up: remove fusion2 to avoid issues when fusions are destroyed
  container->removeFusion(&fusion2);
}

TEST_F(Phase2ContainerTest, MultipleRegistration) {
  // Test multiple Fusions can register with same container
  // Note: We use an IrContainer directly to test the mechanism
  // without interference from Fusion's auto-registration

  auto container = std::make_shared<IrContainer>();

  // Create dummy fusion pointers for testing
  Fusion fusion1;
  Fusion fusion2;
  Fusion fusion3;

  container->addFusion(&fusion1);
  container->addFusion(&fusion2);
  container->addFusion(&fusion3);

  EXPECT_EQ(container->sharingCount(), 3);
  EXPECT_TRUE(container->hasMultipleFusions());

  // Verify all are registered
  const auto& fusions = container->sharingFusions();
  EXPECT_TRUE(fusions.count(&fusion1) > 0);
  EXPECT_TRUE(fusions.count(&fusion2) > 0);
  EXPECT_TRUE(fusions.count(&fusion3) > 0);

  // Clean up: remove all to avoid issues when fusions are destroyed
  container->removeFusion(&fusion1);
  container->removeFusion(&fusion2);
  container->removeFusion(&fusion3);
}

TEST_F(Phase2ContainerTest, StatementCleanup) {
  // Test that removeFusion removes Statements owned by that Fusion
  // This test verifies the cleanup mechanism works through the Fusion lifecycle

  std::shared_ptr<IrContainer> container_ptr;

  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Create some IR
    auto* tv0 = makeSymbolicTensor(2);
    fusion.addInput(tv0);
    auto* tv1 = add(tv0, tv0);
    fusion.addOutput(tv1);

    container_ptr = fusion.ir_container_ptr();

    // With Phase 2, fusion is already registered in constructor
    EXPECT_EQ(container_ptr->sharingCount(), 1);
    EXPECT_GT(container_ptr->vals().size(), 0);
    EXPECT_GT(container_ptr->unordered_exprs().size(), 0);
  }
  // Fusion destroyed - destructor calls removeFusion which cleans up Statements

  // After destruction, count should be 0 and Statements cleaned up
  EXPECT_EQ(container_ptr->sharingCount(), 0);
  EXPECT_EQ(container_ptr->vals().size(), 0);
  EXPECT_EQ(container_ptr->unordered_exprs().size(), 0);
}

// =============================================================================
// Task 3 Tests: Basic shared_ptr Transition
// =============================================================================

TEST_F(Phase2ContainerTest, BasicFusionLifecycle) {
  // Create Fusion, add inputs/outputs, destroy - verify no crashes
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto* tv0 = makeSymbolicTensor(2);
    fusion.addInput(tv0);

    auto* tv1 = add(tv0, tv0);
    fusion.addOutput(tv1);

    // Verify container has expected contents
    EXPECT_GT(fusion.vals().size(), 0);
    EXPECT_GT(fusion.unordered_exprs().size(), 0);
    EXPECT_EQ(fusion.inputs().size(), 1);
    EXPECT_EQ(fusion.outputs().size(), 1);
  }
  // Fusion destroyed here - verify no crashes
  SUCCEED();
}

TEST_F(Phase2ContainerTest, FusionAutoRegistration) {
  // New Fusion automatically registers with its container (sharingCount == 1)
  Fusion fusion;

  auto& container = *fusion.ir_container();

  // With Phase 2 shared_ptr transition, Fusion constructor calls
  // addFusion(this)
  EXPECT_EQ(container.sharingCount(), 1);
  EXPECT_FALSE(container.hasMultipleFusions());

  // The registered Fusion should be our fusion
  const auto& fusions = container.sharingFusions();
  EXPECT_TRUE(fusions.count(&fusion) > 0);
}

TEST_F(Phase2ContainerTest, FusionDestructorCleanup) {
  // Fusion destruction unregisters and cleans up Statements
  std::shared_ptr<IrContainer> container_ptr;

  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Create some IR
    auto* tv0 = makeSymbolicTensor(2);
    fusion.addInput(tv0);
    auto* tv1 = add(tv0, tv0);
    fusion.addOutput(tv1);

    // Capture shared_ptr to container using ir_container_ptr()
    container_ptr = fusion.ir_container_ptr();

    EXPECT_EQ(container_ptr->sharingCount(), 1);
    EXPECT_GT(container_ptr->vals().size(), 0);
    EXPECT_GT(container_ptr->unordered_exprs().size(), 0);
  }
  // Fusion destroyed here - destructor calls removeFusion(this)

  // After Fusion destruction, it should be unregistered
  EXPECT_EQ(container_ptr->sharingCount(), 0);

  // Statements owned by the Fusion should be cleaned up
  EXPECT_EQ(container_ptr->vals().size(), 0);
  EXPECT_EQ(container_ptr->unordered_exprs().size(), 0);
}

TEST_F(Phase2ContainerTest, ContainerAccessor) {
  // Fusion::ir_container_ptr() returns valid shared_ptr
  Fusion fusion;

  // ir_container_ptr() should return a valid shared_ptr
  auto container_ptr = fusion.ir_container_ptr();
  EXPECT_NE(container_ptr, nullptr);

  // The returned shared_ptr should point to the same container as
  // ir_container()
  EXPECT_EQ(container_ptr.get(), fusion.ir_container());

  // We can use the shared_ptr to access container methods
  EXPECT_EQ(container_ptr->sharingCount(), 1);
}

} // namespace nvfuser
