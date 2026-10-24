// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#include "body.h"
#include "core.h"
#include "joint.h"
#include "physics_world.h"
#include "recording.h"
#include "solver.h"
#include "solver_set.h"

// needed for dll export
#include "box3d/box3d.h"

#include <stdio.h>

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

void b3MoverJoint_SetMaxVelocityForce( b3JointId jointId, float maxForce )
{
	B3_ASSERT( b3IsValidFloat( maxForce ) && maxForce >= 0.0f );
	b3World* world = b3GetWorld( jointId.world0 );
	B3_REC( world, MoverJointSetMaxVelocityForce, jointId, maxForce );
	b3JointSim* joint = b3GetJointSimCheckType( jointId, b3_moverJoint );
	joint->moverJoint.maxForce = maxForce;
}

float b3MoverJoint_GetMaxVelocityForce( b3JointId jointId )
{
	b3JointSim* joint = b3GetJointSimCheckType( jointId, b3_moverJoint );
	return joint->moverJoint.maxForce;
}

b3Vec3 b3GetMoverJointForce( b3World* world, b3JointSim* base )
{
	b3MoverJoint* joint = &base->moverJoint;
	b3Vec3 P = b3Blend2( joint->impulse.x, joint->axisX, joint->impulse.y, joint->axisZ );
	return b3MulSV( world->inv_h, P );
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

	b3Quat quatB = b3MulQuat( bodySimB->transform.q, base->localFrameB.q );
	b3Matrix3 matrixB = b3MakeMatrixFromQuat( quatB );
	joint->axisX = matrixB.cx;
	joint->axisZ = matrixB.cz;

	float k = mA + mB;
	joint->linearMass = k > 0.0f ? 1.0f / k : 0.0f;

	if ( context->enableWarmStarting == false )
	{
		joint->impulse = (b3Vec2){ 0 };
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

	b3Vec3 P = b3Blend2( joint->impulse.x, joint->axisX, joint->impulse.y, joint->axisZ );
	if ( stateA->flags & b3_dynamicFlag )
	{
		stateA->linearVelocity = b3MulSub( stateA->linearVelocity, mA, P );
	}

	if ( stateB->flags & b3_dynamicFlag )
	{
		stateB->linearVelocity = b3MulAdd( stateB->linearVelocity, mB, P );
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

	float maxForce = joint->maxForce;
	if ( maxForce > 0.0f )
	{
		b3Vec3 vr = b3Sub( b3Sub( vB, vA ), joint->linearVelocity );
		b3Vec2 cdot = { b3Dot( vr, joint->axisX ), b3Dot( vr, joint->axisZ ) };
		b3Vec2 deltaImpulse = b3MulSV2( -joint->linearMass, cdot );
		b3Vec2 newImpulse = {
			joint->impulse.x + deltaImpulse.x,
			joint->impulse.y + deltaImpulse.y,
		};

		float maxImpulse = context->h * maxForce;

		float lengthSquared = b3Dot2( newImpulse, newImpulse );
		if ( lengthSquared > maxImpulse * maxImpulse )
		{
			float scale = maxImpulse / sqrtf( lengthSquared );
			newImpulse.x *= scale;
			newImpulse.y *= scale;
		}

		deltaImpulse = b3Sub2( newImpulse, joint->impulse );
		joint->impulse = newImpulse;

		b3Vec3 P = b3Blend2( deltaImpulse.x, joint->axisX, deltaImpulse.y, joint->axisZ );
		vA = b3MulSub( vA, mA, P );
		vB = b3MulAdd( vB, mB, P );
	}
	else
	{
		joint->impulse = (b3Vec2){ 0.0f, 0.0f };
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

void b3DrawMoverJoint( b3DebugDraw* draw, b3World* world, b3JointSim* base,
					   b3WorldTransform transformA, b3WorldTransform transformB )
{
	B3_UNUSED( draw, world, base, transformA, transformB );
	B3_ASSERT( base->type == b3_moverJoint );

#if 0
	b3MoverJoint* joint = &base->moverJoint;

	b3Pos pB = b3TransformWorldPoint( transformB, base->localFrameB.p );

	b3Body* bodyA = b3Array_Get( world->bodies, base->bodyIdA );
	b3Body* bodyB = b3Array_Get( world->bodies, base->bodyIdB );
	b3BodyState* stateA = b3GetBodyState( world, bodyA );
	b3BodyState* stateB = b3GetBodyState( world, bodyB );
	b3Vec3 vA = stateA != NULL ? stateA->linearVelocity : b3Vec3_zero;
	b3Vec3 vB = stateB != NULL ? stateB->linearVelocity : b3Vec3_zero;
	b3Vec3 relativeVelocity = b3Sub( vB, vA );

	b3Vec3 maxForce = joint->maxForce;
	b3Vec3 force = b3MulSV( world->inv_h, joint->impulse );

	// todo comment
	float saturation = 0.999f;
	bool saturated = ( maxForce.x > 0.0f && b3AbsFloat( force.x ) >= saturation * maxForce.x ) ||
					 ( maxForce.y > 0.0f && b3AbsFloat( force.y ) >= saturation * maxForce.y ) ||
					 ( maxForce.z > 0.0f && b3AbsFloat( force.z ) >= saturation * maxForce.z );

	// todo comment
	b3Vec3 target = {
		maxForce.x > 0.0f ? joint->linearVelocity.x : relativeVelocity.x,
		maxForce.y > 0.0f ? joint->linearVelocity.y : relativeVelocity.y,
		maxForce.z > 0.0f ? joint->linearVelocity.z : relativeVelocity.z,
	};

	b3HexColor targetColor = saturated ? b3_colorRed : b3_colorLimeGreen;
	b3Pos targetTip = b3OffsetPos( pB, target );
	draw->DrawSegmentFcn( pB, targetTip, targetColor, draw->context );
	draw->DrawPointFcn( targetTip, 6.0f, targetColor, draw->context );

	b3Pos velocityTip = b3OffsetPos( pB, relativeVelocity );
	draw->DrawSegmentFcn( pB, velocityTip, b3_colorLightSkyBlue, draw->context );
	draw->DrawPointFcn( velocityTip, 4.0f, b3_colorLightSkyBlue, draw->context );

	draw->DrawPointFcn( pB, 8.0f, b3_colorPlum, draw->context );

	if ( draw->drawJointExtras )
	{
		if ( bodyA->type != b3_staticBody )
		{
			draw->DrawSegmentFcn( pB, transformA.p, b3_colorDimGray, draw->context );
		}

		b3WorldTransform boxTransform = { pB, b3Quat_identity };
		draw->DrawBoxFcn( b3MulSV( draw->forceScale, maxForce ), boxTransform, b3_colorGray,
						  draw->context );

		b3HexColor forceColor = saturated ? b3_colorRed : b3_colorAzure;
		draw->DrawSegmentFcn( pB, b3OffsetPos( pB, b3MulSV( draw->forceScale, force ) ), forceColor,
							  draw->context );

		char buffer[96];
		snprintf( buffer, 96, "f = (%.1f, %.1f, %.1f)", force.x, force.y, force.z );
		draw->DrawStringFcn( pB, buffer, forceColor, draw->context );
	}

#endif
}
