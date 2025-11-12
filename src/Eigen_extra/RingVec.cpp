#include "RingVec.h"
#include <stdexcept>

namespace EigenExtra {

void RingVec::resize(int capacity) {
	if (capacity < 0) { throw std::invalid_argument("RingVec::resize capacity < 0"); }
	m_capacity = capacity;
	m_buf.assign(static_cast<size_t>(m_capacity), 0.0);
	m_size = 0;
	m_head = 0;
}

void RingVec::clear() {
	if (m_capacity > 0) { std::fill(m_buf.begin(), m_buf.end(), 0.0); }
	m_size = 0;
	m_head = 0;
}

double RingVec::pushPop(double newValue) {
	double old = 0.0;
	if (m_capacity == 0) { return old; }
	if (full()) {
		old = m_buf[static_cast<size_t>(m_head)];
	}
	m_buf[static_cast<size_t>(m_head)] = newValue;
	m_head += 1;
	if (m_head == m_capacity) { m_head = 0; }
	if (m_size < m_capacity) { m_size += 1; }
	return old;
}

int RingVec::logicalToPhysical(int logicalIndex) const {
	if (logicalIndex < 0 || logicalIndex >= m_size) { throw std::out_of_range("RingVec::at index out of range"); }
	// 最旧元素的物理索引：m_head == size 时下一个写入位置，因此最旧在 (m_head - m_size + capacity) % capacity
	int oldest = m_head - m_size;
	if (oldest < 0) { oldest += m_capacity; }
	int pos = oldest + logicalIndex;
	if (pos >= m_capacity) { pos -= m_capacity; }
	return pos;
}

double RingVec::at(int logicalIndex) const {
	if (m_capacity == 0) { throw std::out_of_range("RingVec::at empty buffer"); }
	return m_buf[static_cast<size_t>(logicalToPhysical(logicalIndex))];
}

Ve RingVec::toVe() const {
	Ve out;
	out.resize(m_size);
	if (m_size == 0) { return out; }
	// 逻辑顺序：最旧 -> 最新
	int oldest = m_head - m_size;
	if (oldest < 0) { oldest += m_capacity; }
	const int firstBlock = std::min(m_size, m_capacity - oldest);
	if (firstBlock > 0) {
		Eigen::Map<const Ve> block1(&m_buf[static_cast<size_t>(oldest)], firstBlock);
		out.segment(0, firstBlock) = block1;
	}
	const int remain = m_size - firstBlock;
	if (remain > 0) {
		Eigen::Map<const Ve> block2(m_buf.data(), remain);
		out.segment(firstBlock, remain) = block2;
	}
	return out;
}

void RingVec::assignFromVe(const Ve& sourceVector) {
	if (sourceVector.size() != m_capacity) {
		throw std::invalid_argument("RingVec::assignFromVe size mismatch with capacity");
	}
	if (m_capacity == 0) { return; }
	// 直接写入为满缓冲，逻辑顺序即 sourceVector 的顺序
	m_buf.resize(static_cast<size_t>(m_capacity));
	Eigen::Map<const Ve> src(sourceVector.data(), m_capacity);
	std::copy(src.data(), src.data() + m_capacity, m_buf.data());
	m_size = m_capacity;
	m_head = 0; // 下次写入覆盖最旧：source 的第 0 个
}

std::pair<const double*, int> RingVec::firstSegment() const {
	if (m_size == 0) { return {nullptr, 0}; }
	int oldest = m_head - m_size;
	if (oldest < 0) { oldest += m_capacity; }
	const int firstBlock = std::min(m_size, m_capacity - oldest);
	return { &m_buf[static_cast<size_t>(oldest)], firstBlock };
}

std::pair<const double*, int> RingVec::secondSegment() const {
	if (m_size == 0) { return {nullptr, 0}; }
	int oldest = m_head - m_size;
	if (oldest < 0) { oldest += m_capacity; }
	const int firstBlock = std::min(m_size, m_capacity - oldest);
	const int remain = m_size - firstBlock;
	if (remain <= 0) { return {nullptr, 0}; }
	return { m_buf.data(), remain };
}

} // namespace EigenExtra



