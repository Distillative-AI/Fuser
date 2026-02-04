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

// =============================================================================
// Task 4 Tests: Per-Fusion Statement Tracking
// =============================================================================

TEST_F(Phase2ContainerTest, PerFusionValsTracking) {
  // Test that ownedVals() returns only this Fusion's vals
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create some IR
  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  // ownedVals() should return only this Fusion's vals
  const auto& owned_vals = fusion.ownedVals();
  EXPECT_GT(owned_vals.size(), 0);

  // All vals in ownedVals() should have container() == &fusion
  for (auto* val : owned_vals) {
    EXPECT_EQ(val->container(), &fusion);
  }

  // vals() and ownedVals() should be the same with a single Fusion (Phase 1
  // equivalence)
  EXPECT_EQ(fusion.vals().size(), fusion.ownedVals().size());
}

TEST_F(Phase2ContainerTest, PerFusionExprsTracking) {
  // Test that ownedExprs() returns only this Fusion's exprs
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create some IR
  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  // ownedExprs() should return only this Fusion's exprs
  const auto& owned_exprs = fusion.ownedExprs();
  EXPECT_GT(owned_exprs.size(), 0);

  // All exprs in ownedExprs() should have container() == &fusion
  for (auto* expr : owned_exprs) {
    EXPECT_EQ(expr->container(), &fusion);
  }

  // unordered_exprs() and ownedExprs() should be the same with a single Fusion
  EXPECT_EQ(fusion.unordered_exprs().size(), fusion.ownedExprs().size());
}

TEST_F(Phase2ContainerTest, ValsOwnedByAPI) {
  // Test IrContainer::valsOwnedBy() API directly
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  auto& container = *fusion.ir_container();

  // valsOwnedBy should return same set as ownedVals()
  const auto& vals_by_container = container.valsOwnedBy(&fusion);
  const auto& vals_by_fusion = fusion.ownedVals();
  EXPECT_EQ(vals_by_container.size(), vals_by_fusion.size());

  // valsOwnedBy for a non-registered Fusion should return empty set
  Fusion other_fusion;
  const auto& other_vals = container.valsOwnedBy(&other_fusion);
  EXPECT_EQ(other_vals.size(), 0);
}

TEST_F(Phase2ContainerTest, ExprsOwnedByAPI) {
  // Test IrContainer::exprsOwnedBy() API directly
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  auto& container = *fusion.ir_container();

  // exprsOwnedBy should return same set as ownedExprs()
  const auto& exprs_by_container = container.exprsOwnedBy(&fusion);
  const auto& exprs_by_fusion = fusion.ownedExprs();
  EXPECT_EQ(exprs_by_container.size(), exprs_by_fusion.size());

  // exprsOwnedBy for a non-registered Fusion should return empty set
  Fusion other_fusion;
  const auto& other_exprs = container.exprsOwnedBy(&other_fusion);
  EXPECT_EQ(other_exprs.size(), 0);
}

TEST_F(Phase2ContainerTest, RegisterUpdatesPerFusionTracking) {
  // Test that registering new vals/exprs updates per-Fusion tracking
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Initially no vals
  EXPECT_EQ(fusion.ownedVals().size(), 0);
  EXPECT_EQ(fusion.ownedExprs().size(), 0);

  // Add an input - this creates vals
  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // Now we should have vals tracked for this fusion
  size_t vals_after_input = fusion.ownedVals().size();
  EXPECT_GT(vals_after_input, 0);

  // Add an expression - this creates more vals and exprs
  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  // Both should have grown
  EXPECT_GT(fusion.ownedVals().size(), vals_after_input);
  EXPECT_GT(fusion.ownedExprs().size(), 0);
}

TEST_F(Phase2ContainerTest, TransferStatementOwnership) {
  // Test IrContainer::transferStatementOwnership
  auto container = std::make_shared<IrContainer>();

  // Create dummy Fusions for testing
  Fusion fusion1;
  Fusion fusion2;

  // We can't easily create vals owned by fusion1 in a standalone container,
  // but we can test the tracking data structure directly
  container->addFusion(&fusion1);
  container->addFusion(&fusion2);

  // Transfer ownership - should not crash even with empty tracking
  container->transferStatementOwnership(&fusion1, &fusion2);

  // Verify fusion1 no longer has tracking entries (empty case)
  EXPECT_EQ(container->valsOwnedBy(&fusion1).size(), 0);
  EXPECT_EQ(container->exprsOwnedBy(&fusion1).size(), 0);

  // Cleanup
  container->removeFusion(&fusion1);
  container->removeFusion(&fusion2);
}

TEST_F(Phase2ContainerTest, ClearOnlyAffectsOwnedStatements) {
  // Test that Fusion::clear() only clears THIS Fusion's statements
  // This is critical for shared container correctness

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create some IR
  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  // Get container reference
  auto container_ptr = fusion.ir_container_ptr();

  // Record counts before clear
  size_t vals_before = fusion.ownedVals().size();
  size_t exprs_before = fusion.ownedExprs().size();
  EXPECT_GT(vals_before, 0);
  EXPECT_GT(exprs_before, 0);

  // Clear the fusion
  fusion.clear();

  // After clear, ownedVals/ownedExprs should be empty for this fusion
  EXPECT_EQ(fusion.ownedVals().size(), 0);
  EXPECT_EQ(fusion.ownedExprs().size(), 0);

  // Container-level accessors should also reflect the removal
  EXPECT_EQ(container_ptr->vals().size(), 0);
  EXPECT_EQ(container_ptr->unordered_exprs().size(), 0);
}

TEST_F(Phase2ContainerTest, RemoveStatementsOwnedByAPI) {
  // Test public IrContainer::removeStatementsOwnedBy API
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create some IR
  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  auto& container = *fusion.ir_container();

  // Verify we have statements
  EXPECT_GT(container.vals().size(), 0);
  EXPECT_GT(container.unordered_exprs().size(), 0);
  EXPECT_GT(container.valsOwnedBy(&fusion).size(), 0);
  EXPECT_GT(container.exprsOwnedBy(&fusion).size(), 0);

  // Clear fusion-level state first (inputs_, outputs_, etc.)
  // Note: We're testing the container API directly, not through Fusion::clear()
  // In practice, Fusion::clear() does both
  container.removeStatementsOwnedBy(&fusion);

  // After removal, tracking should be empty
  EXPECT_EQ(container.valsOwnedBy(&fusion).size(), 0);
  EXPECT_EQ(container.exprsOwnedBy(&fusion).size(), 0);

  // Container-level sets should also be empty (single fusion case)
  EXPECT_EQ(container.vals().size(), 0);
  EXPECT_EQ(container.unordered_exprs().size(), 0);
}

// =============================================================================
// Task 7 Tests: Per-Fusion Special Values
// =============================================================================

TEST_F(Phase2ContainerTest, PerFusionSpecialValuesBasic) {
  // Test that special values are created per-Fusion
  Fusion a;
  FusionGuard fg_a(&a);
  Val* zero_a = a.zeroVal();
  Val* one_a = a.oneVal();

  EXPECT_NE(zero_a, nullptr);
  EXPECT_NE(one_a, nullptr);
  EXPECT_EQ(zero_a->container(), &a);
  EXPECT_EQ(one_a->container(), &a);
}

TEST_F(Phase2ContainerTest, SpecialValuesOwnedByFusion) {
  // Test that special values are tracked in ownedVals
  Fusion a;
  FusionGuard fg_a(&a);

  Val* zero_a = a.zeroVal();

  // Special values should be in ownedVals
  EXPECT_TRUE(a.ownedVals().count(zero_a) > 0);
}

TEST_F(Phase2ContainerTest, SeparateFusionsHaveOwnSpecialValues) {
  // Two independent Fusions should have different special values
  Fusion a;
  Fusion b;

  {
    FusionGuard fg_a(&a);
    Val* zero_a = a.zeroVal();
    EXPECT_EQ(zero_a->container(), &a);
  }

  {
    FusionGuard fg_b(&b);
    Val* zero_b = b.zeroVal();
    EXPECT_EQ(zero_b->container(), &b);
  }

  // Each has its own zero (different objects)
  EXPECT_NE(a.zeroVal(), b.zeroVal());
}

TEST_F(Phase2ContainerTest, DestroyFusionDoesNotAffectOther) {
  // Destroying one Fusion should not affect another's special values
  Fusion a;
  FusionGuard fg_a(&a);

  // Create special values in a
  Val* zero_a = a.zeroVal();
  EXPECT_NE(zero_a, nullptr);

  {
    Fusion b;
    FusionGuard fg_b(&b);
    Val* zero_b = b.zeroVal();
    EXPECT_NE(zero_b, nullptr);
    // b destroyed here
  }

  // a should still work fine - its special values should still be valid
  Val* zero_a_again = a.zeroVal();
  EXPECT_EQ(zero_a_again, zero_a);
  EXPECT_EQ(zero_a_again->container(), &a);
}

TEST_F(Phase2ContainerTest, SpecialValuesLazyCreation) {
  // Special values should be created lazily
  Fusion a;
  FusionGuard fg_a(&a);

  // Before calling zeroVal(), it shouldn't exist
  // (Can't directly test this, but we can verify it works after call)
  Val* zero1 = a.zeroVal();
  Val* zero2 = a.zeroVal();

  // Same value returned on repeated calls
  EXPECT_EQ(zero1, zero2);
}

TEST_F(Phase2ContainerTest, AllSpecialValuesPerFusion) {
  // Test all special value accessors
  Fusion a;
  FusionGuard fg_a(&a);

  Val* zero = a.zeroVal();
  Val* one = a.oneVal();
  Val* true_val = a.trueVal();
  Val* false_val = a.falseVal();
  NamedScalar* magic_zero = a.magicZeroVal();

  // All should be non-null
  EXPECT_NE(zero, nullptr);
  EXPECT_NE(one, nullptr);
  EXPECT_NE(true_val, nullptr);
  EXPECT_NE(false_val, nullptr);
  EXPECT_NE(magic_zero, nullptr);

  // All should have container() == &a
  EXPECT_EQ(zero->container(), &a);
  EXPECT_EQ(one->container(), &a);
  EXPECT_EQ(true_val->container(), &a);
  EXPECT_EQ(false_val->container(), &a);
  EXPECT_EQ(magic_zero->container(), &a);

  // All should be tracked in ownedVals
  EXPECT_TRUE(a.ownedVals().count(zero) > 0);
  EXPECT_TRUE(a.ownedVals().count(one) > 0);
  EXPECT_TRUE(a.ownedVals().count(true_val) > 0);
  EXPECT_TRUE(a.ownedVals().count(false_val) > 0);
  EXPECT_TRUE(a.ownedVals().count(magic_zero) > 0);
}

TEST_F(Phase2ContainerTest, SpecialValuesClearedOnFusionClear) {
  // Test that Fusion::clear() resets special values
  Fusion a;
  FusionGuard fg_a(&a);

  // Create special values
  Val* zero_before = a.zeroVal();
  Val* one_before = a.oneVal();
  EXPECT_NE(zero_before, nullptr);
  EXPECT_NE(one_before, nullptr);

  // Clear the fusion
  a.clear();

  // Special values should be recreated lazily (new objects)
  Val* zero_after = a.zeroVal();
  Val* one_after = a.oneVal();

  // The new objects should be different from the old ones
  // (old ones were removed by removeStatementsOwnedBy)
  EXPECT_NE(zero_after, zero_before);
  EXPECT_NE(one_after, one_before);

  // New objects should be valid and owned by the fusion
  EXPECT_EQ(zero_after->container(), &a);
  EXPECT_EQ(one_after->container(), &a);
}

TEST_F(Phase2ContainerTest, SpecialValuesWithDtype) {
  // Test zeroVal(dtype) and oneVal(dtype) accessors
  Fusion a;
  FusionGuard fg_a(&a);

  // Index type should return the cached value
  Val* zero_index = a.zeroVal(DataType::Index);
  Val* zero_cached = a.zeroVal();
  EXPECT_EQ(zero_index, zero_cached);

  Val* one_index = a.oneVal(DataType::Index);
  Val* one_cached = a.oneVal();
  EXPECT_EQ(one_index, one_cached);

  // Bool type should return true/false val
  Val* zero_bool = a.zeroVal(DataType::Bool);
  Val* false_cached = a.falseVal();
  EXPECT_EQ(zero_bool, false_cached);

  Val* one_bool = a.oneVal(DataType::Bool);
  Val* true_cached = a.trueVal();
  EXPECT_EQ(one_bool, true_cached);

  // Other types should create new values (not cached)
  Val* zero_float = a.zeroVal(DataType::Float);
  Val* zero_float2 = a.zeroVal(DataType::Float);
  // These are not cached, so they're different objects
  EXPECT_NE(zero_float, zero_float2);
}

} // namespace nvfuser
