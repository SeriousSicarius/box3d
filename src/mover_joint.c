// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#include "body.h"
#include "core.h"
#include "joint.h"
#include "math_internal.h"
#include "physics_world.h"
#include "recording.h"
#include "solver.h"
#include "solver_set.h"

// needed for dll export
#include "box3d/box3d.h"

void b3MoverJoint_SetLinearVelocity( b3JointId jointId, b3Vec3 velocity )
{
	B3_ASSERT( b3IsValidVec3( velocity ) );
	b3World* world = b3GetWorld( jointId.world0 );
	B3_REC( world, MoverJointSetLinearVelocity, jointId, velocity );
	b3JointSim* joint = b3GetJointSimCheckType( jointId, b3_moverJoint );
	joint->moverJoint.linearVelocity = velocity;
}

b3Vec3 b3MoverJoint_GetLinearVelocity( b3JointId jointId )
{
	b3JointSim* joint = b3GetJointSimCheckType( jointId, b3_moverJoint );
	return joint->moverJoint.linearVelocity;
}

void b3MoverJoint_SetMaxVelocityForce( b3JointId jointId, b3Vec3 maxForce )
{
	B3_ASSERT( b3IsValidVec3( maxForce ) );
	b3World* world = b3GetWorld( jointId.world0 );
	B3_REC( world, MoverJointSetMaxVelocityForce, jointId, maxForce );
	b3JointSim* joint = b3GetJointSimCheckType( jointId, b3_moverJoint );
	joint->moverJoint.maxVelocityForce = maxForce;
}

b3Vec3 b3MoverJoint_GetMaxVelocityForce( b3JointId jointId )
{
	b3JointSim* joint = b3GetJointSimCheckType( jointId, b3_moverJoint );
	return joint->moverJoint.maxVelocityForce;
}

b3Vec3 b3GetMoverJointForce( b3World* world, b3JointSim* base )
{
	return b3MulSV( world->inv_h, base->moverJoint.linearVelocityImpulse );
}

void b3PrepareMoverJoint( b3JointSim* base, b3StepContext* context )
{
	B3_ASSERT( base->type == b3_moverJoint );

	int idA = base->bodyIdA;
	int idB = base->bodyIdB;

	b3World* world = context->world;

	b3Body* bodyA = b3Array_Get( world->bodies, idA );
	b3Body* bodyB = b3Array_Get( world->bodies, idB );

	B3_ASSERT( bodyA->setIndex == b3_awakeSet || bodyB->setIndex == b3_awakeSet );

	b3SolverSet* setA = b3Array_Get( world->solverSets, bodyA->setIndex );
	b3SolverSet* setB = b3Array_Get( world->solverSets, bodyB->setIndex );

	int localIndexA = bodyA->localIndex;
	int localIndexB = bodyB->localIndex;

	b3BodySim* bodySimA = b3Array_Get( setA->bodySims, localIndexA );
	b3BodySim* bodySimB = b3Array_Get( setB->bodySims, localIndexB );

	float mA = bodySimA->invMass;
	float mB = bodySimB->invMass;

	base->invMassA = mA;
	base->invMassB = mB;
	base->invIA = b3Mat3_zero;
	base->invIB = b3Mat3_zero;

	b3MoverJoint* joint = &base->moverJoint;
	joint->indexA = bodyA->setIndex == b3_awakeSet ? localIndexA : B3_NULL_INDEX;
	joint->indexB = bodyB->setIndex == b3_awakeSet ? localIndexB : B3_NULL_INDEX;

	float k = mA + mB;
	joint->linearMass = k > 0.0f ? 1.0f / k : 0.0f;

	if ( context->enableWarmStarting == false )
	{
		joint->linearVelocityImpulse = b3Vec3_zero;
	}
}

void b3WarmStartMoverJoint( b3JointSim* base, b3StepContext* context )
{
	B3_ASSERT( base->type == b3_moverJoint );

	float mA = base->invMassA;
	float mB = base->invMassB;

	b3MoverJoint* joint = &base->moverJoint;

	// dummy state for static bodies
	b3BodyState dummyState = b3_identityBodyState;

	b3BodyState* stateA =
		joint->indexA == B3_NULL_INDEX ? &dummyState : context->states + joint->indexA;
	b3BodyState* stateB =
		joint->indexB == B3_NULL_INDEX ? &dummyState : context->states + joint->indexB;

	if ( stateA->flags & b3_dynamicFlag )
	{
		stateA->linearVelocity =
			b3MulSub( stateA->linearVelocity, mA, joint->linearVelocityImpulse );
	}

	if ( stateB->flags & b3_dynamicFlag )
	{
		stateB->linearVelocity =
			b3MulAdd( stateB->linearVelocity, mB, joint->linearVelocityImpulse );
	}
}

void b3SolveMoverJoint( b3JointSim* base, b3StepContext* context )
{
	B3_ASSERT( base->type == b3_moverJoint );

	float mA = base->invMassA;
	float mB = base->invMassB;

	// dummy state for static bodies
	b3BodyState dummyState = b3_identityBodyState;

	b3MoverJoint* joint = &base->moverJoint;
	b3BodyState* stateA =
		joint->indexA == B3_NULL_INDEX ? &dummyState : context->states + joint->indexA;
	b3BodyState* stateB =
		joint->indexB == B3_NULL_INDEX ? &dummyState : context->states + joint->indexB;

	b3Vec3 vA = stateA->linearVelocity;
	b3Vec3 vB = stateB->linearVelocity;

	b3Vec3 maxForce = joint->maxVelocityForce;
	if ( maxForce.x > 0.0f || maxForce.y > 0.0f || maxForce.z > 0.0f )
	{
		b3Vec3 cdot = b3Sub( b3Sub( vB, vA ), joint->linearVelocity );
		b3Vec3 impulse = b3MulSV( -joint->linearMass, cdot );

		b3Vec3 oldImpulse = joint->linearVelocityImpulse;
		b3Vec3 maxImpulse = b3MulSV( context->h, maxForce );
		joint->linearVelocityImpulse =
			b3Clamp( b3Add( oldImpulse, impulse ), b3Neg( maxImpulse ), maxImpulse );
		impulse = b3Sub( joint->linearVelocityImpulse, oldImpulse );

		vA = b3MulSub( vA, mA, impulse );
		vB = b3MulAdd( vB, mB, impulse );
	}
	else
	{
		joint->linearVelocityImpulse = b3Vec3_zero;
	}

	if ( stateA->flags & b3_dynamicFlag )
	{
		stateA->linearVelocity = vA;
	}

	if ( stateB->flags & b3_dynamicFlag )
	{
		stateB->linearVelocity = vB;
	}
}
