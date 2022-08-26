//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {

    /// create world
    world_ = std::make_unique<raisim::World>();
    // heightMap_ = world_->addHeightMap("/home/seok-ju/raisim_ws/raisimLib/rsc/xmlScripts/heightMaps/zurichHeightMap.png", 0, 0, 500, 500, 0.005, -10);
    /// add objects
    anymal_ = world_->addArticulatedSystem(resourceDir_+"/anymal_c_arm/urdf/anymal_c_arm.urdf");
    anymal_->setName("anymal");
    anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);
    ranJoint_.setZero(6);
    
    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, -0.4, 0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, 0.0, 1.2, 1.8, 0.0, 0.8, 0.0;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);

    jointPgain.setZero(); jointPgain.tail(6) << 160.0, 160.0, 160.0, 45.0, 45.0, 45.0;
    jointDgain.setZero(); jointDgain.tail(6) << 6.0, 6.0, 6.0, 1.5, 1.5, 1.5;
    jointPgain.segment(7, nJoints_-6).setConstant(50.0);
    jointDgain.segment(7, nJoints_-6).setConstant(0.2);
    
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 46;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.3);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(anymal_);
    }
  }

  void init() final { }

  void reset() final {
    anymal_->setState(gc_init_, gv_init_);
    updateObservation();
  }

  float stepran(const Eigen::Ref<EigenVec>& action, const Eigen::Ref<EigenVec>& random_sample) {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    anymal_->setPdTarget(pTarget_, vTarget_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    updateObservation();

    ranval_ = random_sample.cast<double>();
    ranJoint_ = ranval_.tail(6);

    curjoint_ = gc_.tail(6);
    // error_ = (ranJoint_ - curjoint_).squaredNorm();
    force_.setZero(12);
    for (int i=0; i<12; i++){
      force_[i] = anymal_->getGeneralizedForce()[6+i];
    }

    rewards_.record("torque", force_.squaredNorm());
    if (ranval_[0] >= 0) {
      rewards_.record("forwardVel", std::min(ranval_[0], bodyLinearVel_[0]));
    }
    // else if (ranval_[0] == 0) {
    //   rewards_.record("zeroVel", exp(-abs(bodyLinearVel_[0])));
    // }
    else {
      rewards_.record("backwardVel", std::max(ranval_[0], bodyLinearVel_[0]));
    }
    if (ranval_[1] >= 0) {
      rewards_.record("rightwardVel", std::min(ranval_[1], bodyLinearVel_[1]));
    }
    // else if (ranval_[1] == 0) {
    //   rewards_.record("zeroVel", exp(-abs(bodyLinearVel_[1])));
    // }
    else {
      rewards_.record("leftwardVel", std::max(ranval_[1], bodyLinearVel_[1]));
    }
    if (ranval_[2] >= 0) {
      rewards_.record("ccwiseVel", std::min(ranval_[2], bodyAngularVel_[2]));
    }
    // else if (ranval_[2] == 0) {
    //   rewards_.record("zeroVel", exp(-abs(bodyAngularVel_[2])));
    // }
    else {
      rewards_.record("cwiseVel", std::max(ranval_[2], bodyAngularVel_[2]));
    }
    for (int i=0; i<6; i++){
      rewards_.record("ccwisejoint", exp(-abs(curjoint_[i]-(ranval_[3+i]+gc_init_[19+i]))));
    }
    // rewards_.record("joint", exp(-error_));

    return rewards_.sum();
  }

  void updateObservation() {
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    obDouble_ << gc_[2], /// body height
        rot.e().row(2).transpose(), /// body orientation
        gc_.tail(18), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(18); /// joint velocity
        // ranval_,
        // ranJoint_;
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  void testob(Eigen::Ref<EigenVec> ob, float& ran_x, float& ran_y, float& ran_yaw, float& j1, float& j2, float& j3, float& j4, float& j5, float& j6) {
    /// convert it to float
    ob = obDouble_.cast<float>();
    ob.tail(9) << ran_x,
                  ran_y,
                  ran_yaw,
                  j1,
                  j2,
                  j3,
                  j4,
                  j5,
                  j6;
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for(auto& contact: anymal_->getContacts())
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
        return true;

    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() { };

 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* anymal_;
  raisim::HeightMap* heightMap_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, ranTar_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_, rewards_set_, curjoint_, ranJoint_, ranval_, force_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  raisim::Vec<3> ee_Vel_;
  raisim::Mat<3,3> ee_Ori_;
  raisim::Vec<4> ee_quat_;
  std::set<size_t> footIndices_;
  float error_;
  float ranx_, rany_, ranyaw_;

  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

