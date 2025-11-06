// contracts/FLRegistry.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";

contract FLRegistry is AccessControl {
    bytes32 public constant CLIENT_ROLE = keccak256("CLIENT_ROLE");
    bytes32 public constant AUDITOR_ROLE = keccak256("AUDITOR_ROLE");

    event ClientRegistered(address indexed client);
    event ClientRevoked(address indexed client);
    event CommitRecorded(
        uint256 indexed roundId,
        address indexed client,
        string updateHash,
        int256 dpEpsilonScaled,   // epsilon * 1e6 (أو -1 إذا غير مفعل)
        uint256 numSamples
    );
    event RoundFinalized(
        uint256 indexed roundId,
        string globalHash,
        uint256 participating,
        string metricsJson
    );
    event DataAccess(
        address indexed actor,
        string patientHash,       // hash(id/pseudonym)
        string purposeHash,       // hash(purpose)
        uint256 ts
    );

    // عتبة اختيارية (تنظيمية) للخصوصية التفاضلية
    int256 public minEpsilonScaled = -1; // -1 = غير مفعّلة

    constructor(address admin) {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(AUDITOR_ROLE, admin);
    }

    function registerClient(address client) external onlyRole(DEFAULT_ADMIN_ROLE) {
        _grantRole(CLIENT_ROLE, client);
        emit ClientRegistered(client);
    }

    function revokeClient(address client) external onlyRole(DEFAULT_ADMIN_ROLE) {
        _revokeRole(CLIENT_ROLE, client);
        emit ClientRevoked(client);
    }

    function setMinEpsilonScaled(int256 epsScaled) external onlyRole(DEFAULT_ADMIN_ROLE) {
        minEpsilonScaled = epsScaled; // مثال: 500000 => ε=0.5
    }

    function recordClientCommit(
        uint256 roundId,
        string calldata updateHash,
        int256 dpEpsilonScaled,
        uint256 numSamples
    ) external onlyRole(CLIENT_ROLE) {
        // سياسة بسيطة: لو تم ضبط minEpsilonScaled، تحقّق
        if (minEpsilonScaled >= 0 && dpEpsilonScaled >= 0) {
            require(dpEpsilonScaled <= minEpsilonScaled, "DP epsilon above policy");
        }
        emit CommitRecorded(roundId, msg.sender, updateHash, dpEpsilonScaled, numSamples);
    }

    function recordRoundFinalize(
        uint256 roundId,
        string calldata globalHash,
        uint256 participating,
        string calldata metricsJson
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        emit RoundFinalized(roundId, globalHash, participating, metricsJson);
    }

    function recordDataAccess(
        string calldata patientHash,
        string calldata purposeHash
    ) external onlyRole(CLIENT_ROLE) {
        emit DataAccess(msg.sender, patientHash, purposeHash, block.timestamp);
    }
}
